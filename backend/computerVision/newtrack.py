"""
Basketball Tracker with AWS SageMaker Integration
- Only boxes humans (filters walls/floor)
- Clusters all detected persons into exactly 2 teams using per-player embeddings
- Team 1: RED, Team 2: BLUE
- Ball: ORANGE
- Hardcoded net/hoop (normalized coords)
- Saves annotated frames to disk (no video)
"""

import cv2
import boto3
import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import os
from pathlib import Path

class BasketballTrackerSageMaker:
    def __init__(self, endpoint_name, region='us-east-1', confidence=0.3):
        print(f"Connecting to SageMaker endpoint: {endpoint_name}")
        self.endpoint_name = endpoint_name
        self.confidence = confidence

        # SageMaker runtime
        self.sagemaker_client = boto3.client('runtime.sagemaker', region_name=region)

        # persistent per-track embeddings (rolling window)
        self.player_embeddings = defaultdict(list)

        # tracking
        self.player_tracker = {}   # maps bbox tuple -> track_id
        self.next_track_id = 0

        # fps stats
        self.fps = 0.0
        self.frame_times = []

        # Team assignments (track_id -> 0 or 1)
        self.team_assignments = {}

        # Colors: BGR
        self.team_colors = {0: (0, 0, 255),   # Team 1: RED
                            1: (255, 0, 0)}   # Team 2: BLUE
        self.ball_color = (0, 165, 255)       # ORANGE (BGR)

        # Hardcoded hoop / net in normalized coordinates (left, top, right, bottom)
        # These were picked based on your JumpStart outputs (top-right-ish). Adjust if needed.
        # We'll convert to pixels per frame when drawing.
        self.hoop_bbox_norm = (0.5, 0.02, 0.65, 0.32)

        print("SageMaker tracker initialized!")

    def query_sagemaker_endpoint(self, frame):
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        try:
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/x-image',
                Body=img_bytes,
                Accept='application/json'
            )
            result = json.loads(response['Body'].read())
            # debug: prints first 1000 chars
            print("SageMaker raw output:", json.dumps(result, indent=2)[:1000])
            return self.parse_sagemaker_response(result, frame.shape)
        except Exception as e:
            print(f"SageMaker inference error: {e}")
            return {'players': [], 'balls': []}

    def parse_sagemaker_response(self, response, frame_shape):
        """
        Robust parsing for JumpStart-style responses:
        - If only 'normalized_boxes' returned, treat boxes as object proposals.
        - Filter proposals to keep probable people (filter out extremely large boxes / tiny boxes, etc.)
        - Heuristics to detect ball (tiny nearly-square boxes)
        """
        detections = {'players': [], 'balls': []}
        h, w = frame_shape[:2]

        # Helper: normalized -> pixel
        def norm_to_pixels(box):
            l, t, r, b = box
            x1 = int(max(0, min(1, l)) * w)
            y1 = int(max(0, min(1, t)) * h)
            x2 = int(max(0, min(1, r)) * w)
            y2 = int(max(0, min(1, b)) * h)
            return [x1, y1, x2, y2]

        def is_likely_human(norm_box):
            l, t, r, b = norm_box
            width = max(0.0, r - l)
            height = max(0.0, b - t)
            area = width * height

            # skip obvious non-humans
            if width > 0.9 or height > 0.95 or area > 0.6:
                return False
            if width / (height + 1e-6) > 0.8:  # must be taller than wide
                return False
            if height < 0.05 or width < 0.02:
                return False
            if area < 0.01:  # skip tiny objects (like balls)
                return False
            if t >= b:
                return False
            return True

        def is_likely_ball(norm_box):
            l, t, r, b = norm_box
            width = max(0.0, r - l)
            height = max(0.0, b - t)
            area = width * height
            ratio = width / (height + 1e-6)

            # smaller and roughly square
            if area < 0.0005 or area > 0.02:
                return False
            if 0.75 <= ratio <= 1.25:
                return True
            return False


        # If normalized_boxes provided (common JumpStart output), parse them:
        if 'normalized_boxes' in response:
            boxes = response['normalized_boxes']
            # If there are classes / scores provided align them; else use heuristics
            classes = response.get('classes', [])
            scores = response.get('scores', [])

            for i, box in enumerate(boxes):
                # skip malformed
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                # Some boxes might have coords out of [0,1], still normalize clamp above
                norm_box = (box[0], box[1], box[2], box[3])

                # If model provided explicit classes/scores use them when possible
                score = None
                cls = None
                if i < len(scores):
                    try:
                        score = float(scores[i])
                    except Exception:
                        score = None
                if i < len(classes):
                    try:
                        cls = int(classes[i])
                    except Exception:
                        cls = None

                # If score exists and below confidence -> skip
                if score is not None and score < self.confidence:
                    continue

                # If a class id is present and it clearly maps to "person" by id=0 (common)
                # we use that. If not, fall back to heuristics.
                pixel_bbox = norm_to_pixels(norm_box)
                area_norm = (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1])

                # Heuristic ball detection
                if is_likely_ball(norm_box):
                    detections['balls'].append({'bbox': pixel_bbox, 'confidence': float(score if score is not None else 1.0)})
                    continue

                # Keep box only if it's likely a human
                if is_likely_human(norm_box):
                    detections['players'].append({'bbox': pixel_bbox, 'confidence': float(score if score is not None else 1.0)})
                else:
                    # not a human by heuristics -> skip
                    continue

        # If predictions field present (alternate format)
        elif 'predictions' in response:
            for pred in response['predictions']:
                bbox = pred.get('bbox', pred.get('bounding_box'))
                if not bbox or len(bbox) < 4:
                    continue
                # Assume bbox normalized or absolute; if values <=1 treat normalized
                if max(bbox) <= 1.1:
                    norm_box = (bbox[0], bbox[1], bbox[2], bbox[3])
                    pixel_bbox = norm_to_pixels(norm_box)
                else:
                    # already pixel coords
                    pixel_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                conf = float(pred.get('confidence', pred.get('score', 1.0)))
                label = str(pred.get('class', pred.get('label', ''))).lower()

                if conf < self.confidence:
                    continue

                if 'person' in label or 'player' in label:
                    detections['players'].append({'bbox': pixel_bbox, 'confidence': conf})
                elif 'ball' in label or 'basket' in label:
                    detections['balls'].append({'bbox': pixel_bbox, 'confidence': conf})

        return detections

    def simple_track(self, current_bboxes, prev_tracker, max_distance=60):
        """
        Track by greedy matching centroid distances.
        Inputs: current_bboxes = list of [x1,y1,x2,y2]
        prev_tracker: dict {bbox_tuple: track_id}
        """
        new_tracker = {}
        used_prev = set()

        # precompute centers
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in current_bboxes]
        prev_items = list(prev_tracker.items())  # (prev_bbox_tuple, track_id)

        for i, bbox in enumerate(current_bboxes):
            cx, cy = centers[i]
            best_id = None
            best_dist = float('inf')
            best_prev = None

            for prev_bbox, tid in prev_items:
                if prev_bbox in used_prev:
                    continue
                px1, py1, px2, py2 = prev_bbox
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                d = np.hypot(cx - pcx, cy - pcy)
                if d < best_dist and d < max_distance:
                    best_dist = d
                    best_id = tid
                    best_prev = prev_bbox

            if best_id is not None:
                new_tracker[tuple(bbox)] = best_id
                used_prev.add(best_prev)
            else:
                new_tracker[tuple(bbox)] = self.next_track_id
                self.next_track_id += 1

        return new_tracker

    def extract_color_histogram(self, image, bbox):
        """Return 48-d color histogram embedding from center crop of bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(48, dtype=float)

        crop = image[y1:y2, x1:x2]
        # center crop safely
        ch, cw = crop.shape[:2]
        a_h = max(1, int(ch * 0.3)); b_h = max(1, int(ch * 0.7))
        a_w = max(1, int(cw * 0.3)); b_w = max(1, int(cw * 0.7))
        center_crop = crop[a_h:b_h, a_w:b_w]
        if center_crop.size == 0:
            center_crop = crop

        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        hist = np.concatenate([cv2.normalize(hist_h, hist_h).flatten(),
                               cv2.normalize(hist_s, hist_s).flatten(),
                               cv2.normalize(hist_v, hist_v).flatten()])
        return hist

    def cluster_teams(self):
        """
        Cluster all currently tracked players into 2 teams using the average embedding
        for each track. Always outputs exactly two labels {0,1} for each tracked id.
        """
        # compute mean embedding for each track id
        track_ids = list(self.player_embeddings.keys())
        if not track_ids:
            self.team_assignments = {}
            return

        avg_embs = []
        ids = []
        for tid in track_ids:
            embs = np.array(self.player_embeddings[tid])
            if embs.size == 0:
                # create a small random vector so KMeans still works robustly
                avg_emb = np.zeros(48)
            else:
                avg_emb = np.mean(embs, axis=0)
            avg_embs.append(avg_emb)
            ids.append(tid)

        avg_embs = np.array(avg_embs)
        # dimensionality reduction if needed
        if avg_embs.shape[1] > 10 and avg_embs.shape[0] > 1:
            n_components = min(10, avg_embs.shape[0], avg_embs.shape[1])
            pca = PCA(n_components=n_components)
            avg_embs_reduced = pca.fit_transform(avg_embs)
        else:
            avg_embs_reduced = avg_embs

        # If only one player, assign them to team 0
        if avg_embs_reduced.shape[0] == 1:
            self.team_assignments = {ids[0]: 0}
            return

        # run kmeans into exactly two clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(avg_embs_reduced)

        # assign team labels
        assignments = {}
        for tid, lab in zip(ids, labels):
            assignments[tid] = int(lab)  # 0 or 1

        # store assignments (overwrite)
        self.team_assignments = assignments

    def process_frame(self, frame, frame_count, cluster_interval=30):
        # FPS update
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

        # inference
        detections = self.query_sagemaker_endpoint(frame)

        # track players
        player_bboxes = [d['bbox'] for d in detections['players']]
        self.player_tracker = self.simple_track(player_bboxes, self.player_tracker, max_distance=60)

        # update embeddings per track (rolling window)
        for bbox_tuple, track_id in self.player_tracker.items():
            bbox = list(bbox_tuple)
            emb = self.extract_color_histogram(frame, bbox)
            self.player_embeddings[track_id].append(emb)
            if len(self.player_embeddings[track_id]) > 60:
                self.player_embeddings[track_id].pop(0)

        # run clustering every cluster_interval frames (or when new players appear)
        if frame_count % cluster_interval == 0:
            self.cluster_teams()

        # Annotate
        annotated = frame.copy()

        # draw hardcoded hoop in pixels
        h, w = frame.shape[:2]
        hl, ht, hr, hb = self.hoop_bbox_norm
        hx1 = int(max(0, hl) * w)
        hy1 = int(max(0, ht) * h)
        hx2 = int(min(1.0, hr) * w)
        hy2 = int(min(1.0, hb) * h)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
        cv2.putText(annotated, "NET", (hx1, max(0, hy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Draw players with their team colors (force assignment)
        for bbox_tuple, track_id in self.player_tracker.items():
            bbox = list(map(int, bbox_tuple))
            team = self.team_assignments.get(track_id, 0)  # default 0 if not yet clustered
            color = self.team_colors[team]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"T{team+1} ID:{track_id}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw balls (orange)
        for ball in detections['balls']:
            bx1, by1, bx2, by2 = map(int, ball['bbox'])
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), self.ball_color, 2)
            cv2.putText(annotated, "BALL", (bx1, max(0, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_color, 2)

        # stats overlay
        team_counts = defaultdict(int)
        for t in self.team_assignments.values():
            team_counts[t] += 1

        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (420, 120), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated, f"Team1(Red): {team_counts.get(0,0)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.team_colors[0], 2)
        cv2.putText(annotated, f"Team2(Blue): {team_counts.get(1,0)}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.team_colors[1], 2)

        return annotated

    def process_frames_directory(self, frames_dir, output_dir='backend/computerVision/annotated_frames', fps=30):
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob('*.jpg')) + sorted(frames_path.glob('*.png'))
        if not frame_files:
            print(f"ERROR: No frames found in {frames_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"Found {len(frame_files)} frames in {frames_dir}. Saving annotated frames to: {output_dir}")

        for frame_count, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"WARNING: Could not read {frame_file}, skipping...")
                continue

            annotated = self.process_frame(frame, frame_count)
            out_path = os.path.join(output_dir, frame_file.name)  # keep original filename
            cv2.imwrite(out_path, annotated)

            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{len(frame_files)} frames...")

        print("\n=== Frame Processing Complete ===")
        print(f"Total frames processed: {len(frame_files)}")
        print(f"Annotated frames saved to: {output_dir}")


def main():
    ENDPOINT_NAME = 'jumpstart-dft-mx-od-yolo3-mobilenet-20251025-183254'
    AWS_REGION = "us-west-2"

    tracker = BasketballTrackerSageMaker(endpoint_name=ENDPOINT_NAME, region=AWS_REGION, confidence=0.3)
    tracker.process_frames_directory(
        frames_dir=r"backend\computerVision\frames",
        output_dir=r"backend\computerVision\annotated_frames",
        fps=30
    )

if __name__ == "__main__":
    main()