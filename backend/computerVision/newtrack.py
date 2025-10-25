"""
Basketball Tracker with AWS SageMaker Integration
Detects players and ball using SageMaker endpoint, then applies team clustering locally
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

class BasketballTrackerSageMaker:
    def __init__(self, endpoint_name, region='us-east-1', confidence=0.3):
        """
        Initialize basketball tracker with SageMaker integration
        
        Args:
            endpoint_name: Name of your deployed SageMaker endpoint
            region: AWS region
            confidence: Detection confidence threshold
        """
        print(f"Connecting to SageMaker endpoint: {endpoint_name}")
        self.endpoint_name = endpoint_name
        self.confidence = confidence
        
        # Initialize SageMaker runtime client
        self.sagemaker_client = boto3.client('runtime.sagemaker', region_name=region)
        
        # Team assignment storage
        self.team_assignments = {}
        self.player_embeddings = defaultdict(list)
        
        # Team colors (BGR format for OpenCV)
        self.team_colors = {
            0: (255, 0, 0),    # Blue for Team 1
            1: (0, 0, 255),    # Red for Team 2
            -1: (128, 128, 128) # Gray for unassigned
        }
        
        # Tracking
        self.player_tracker = {}  # Simple tracker: bbox -> track_id
        self.next_track_id = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_times = []
        
        # Hoop tracking
        self.hoop_bbox = None
        self.team_shots = {0: 0, 1: 0}
        self.ball_possession = None
        
        print("SageMaker tracker initialized!")
    
    def query_sagemaker_endpoint(self, frame):
        """
        Send frame to SageMaker endpoint for inference
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            Parsed detections
        """
        # Encode frame as JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        try:
            # Invoke SageMaker endpoint
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/x-image',
                Body=img_bytes,
                Accept='application/json'
            )
            
            # Parse response
            result = json.loads(response['Body'].read())
            print("SageMaker raw output:", json.dumps(result, indent=2)[:1000])
            return self.parse_sagemaker_response(result, frame.shape)
            
        except Exception as e:
            print(f"SageMaker inference error: {e}")
            return {'players': [], 'balls': []}
    
    def parse_sagemaker_response(self, response, frame_shape):
        """
        Parse SageMaker response into player and ball detections
        
        Args:
            response: SageMaker endpoint response
            frame_shape: (height, width, channels)
            
        Returns:
            Dictionary with players and balls
        """
        detections = {'players': [], 'balls': []}
        
        h, w = frame_shape[:2]
        
        # Parse based on response format
        # Adjust this based on your SageMaker model's output format
        if 'normalized_boxes' in response:
            boxes = response['normalized_boxes']
            classes = response.get('classes', [])
            scores = response.get('scores', [])
            labels = response.get('labels', {})
            
            for i, box in enumerate(boxes):
                if i >= len(scores) or scores[i] < self.confidence:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                left, bot, right, top = box
                x1 = int(left * w)
                y1 = int(bot * h)
                x2 = int(right * w)
                y2 = int(top * h)
                
                bbox = [x1, y1, x2, y2]
                conf = scores[i]
                
                # Get class name
                class_idx = int(classes[i]) if i < len(classes) else -1
                class_name = labels.get(str(class_idx), 'unknown')
                
                # Classify as player or ball
                if 'person' in class_name.lower() or 'player' in class_name.lower():
                    detections['players'].append({'bbox': bbox, 'confidence': conf})
                elif 'ball' in class_name.lower() or 'basketball' in class_name.lower():
                    detections['balls'].append({'bbox': bbox, 'confidence': conf})
        
        # Alternative format: 'predictions' or 'detections'
        elif 'predictions' in response:
            for pred in response['predictions']:
                bbox = pred.get('bbox', pred.get('bounding_box'))
                conf = pred.get('confidence', pred.get('score', 1.0))
                class_name = pred.get('class', pred.get('label', ''))
                
                if conf < self.confidence:
                    continue
                
                if 'person' in class_name.lower() or 'player' in class_name.lower():
                    detections['players'].append({'bbox': bbox, 'confidence': conf})
                elif 'ball' in class_name.lower():
                    detections['balls'].append({'bbox': bbox, 'confidence': conf})
        
        return detections
    
    def simple_track(self, current_bboxes, prev_tracker, max_distance=50):
        """
        Simple tracking by matching boxes with closest previous boxes
        
        Args:
            current_bboxes: List of current frame bboxes
            prev_tracker: Previous frame tracker dict
            max_distance: Maximum distance for matching
            
        Returns:
            Dictionary mapping bbox index to track_id
        """
        new_tracker = {}
        
        for i, bbox in enumerate(current_bboxes):
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Find closest previous detection
            min_dist = float('inf')
            matched_id = None
            
            for prev_bbox, track_id in prev_tracker.items():
                px1, py1, px2, py2 = prev_bbox
                prev_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                
                dist = np.sqrt((center[0] - prev_center[0])**2 + 
                              (center[1] - prev_center[1])**2)
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = track_id
            
            # Assign track ID
            if matched_id is not None:
                new_tracker[tuple(bbox)] = matched_id
            else:
                new_tracker[tuple(bbox)] = self.next_track_id
                self.next_track_id += 1
        
        return new_tracker
    
    def extract_color_histogram(self, image, bbox):
        """Extract color histogram from player crop"""
        x1, y1, x2, y2 = map(int, bbox)
        
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(48)
        
        # Center crop
        h_crop, w_crop = crop.shape[:2]
        center_h = int(h_crop * 0.3), int(h_crop * 0.7)
        center_w = int(w_crop * 0.3), int(w_crop * 0.7)
        center_crop = crop[center_h[0]:center_h[1], center_w[0]:center_w[1]]
        
        if center_crop.size == 0:
            center_crop = crop
        
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def cluster_teams(self, embeddings_dict, min_samples=5):
        """Cluster players into teams"""
        valid_ids = [tid for tid, embs in embeddings_dict.items() 
                     if len(embs) >= min_samples]
        
        if len(valid_ids) < 2:
            return {valid_ids[0]: 0} if len(valid_ids) == 1 else {}
        
        avg_embeddings = []
        for tid in valid_ids:
            embs = np.array(embeddings_dict[tid])
            avg_embeddings.append(np.mean(embs, axis=0))
        
        avg_embeddings = np.array(avg_embeddings)
        
        n_samples = avg_embeddings.shape[0]
        n_features = avg_embeddings.shape[1]
        
        if n_features > 10 and n_samples > 10:
            pca = PCA(n_components=10)
            avg_embeddings = pca.fit_transform(avg_embeddings)
        elif n_features > n_samples:
            pca = PCA(n_components=min(n_samples - 1, n_features))
            avg_embeddings = pca.fit_transform(avg_embeddings)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(avg_embeddings)
        
        return {tid: int(label) for tid, label in zip(valid_ids, team_labels)}
    
    def process_frame(self, frame, frame_count, cluster_interval=30):
        """Process frame with SageMaker inference"""
        
        # Update FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        
        # Get detections from SageMaker
        detections = self.query_sagemaker_endpoint(frame)
        
        # Track players
        player_bboxes = [d['bbox'] for d in detections['players']]
        self.player_tracker = self.simple_track(player_bboxes, self.player_tracker)
        
        # Extract embeddings for clustering
        for bbox_tuple, track_id in self.player_tracker.items():
            bbox = list(bbox_tuple)
            embedding = self.extract_color_histogram(frame, bbox)
            self.player_embeddings[track_id].append(embedding)
            
            if len(self.player_embeddings[track_id]) > 50:
                self.player_embeddings[track_id].pop(0)
        
        # Periodically cluster teams
        if frame_count % cluster_interval == 0 and len(self.player_embeddings) > 0:
            self.team_assignments = self.cluster_teams(self.player_embeddings)
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw players with team colors
        for bbox_tuple, track_id in self.player_tracker.items():
            team = self.team_assignments.get(track_id, -1)
            color = self.team_colors[team]
            
            x1, y1, x2, y2 = map(int, bbox_tuple)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            team_name = f"Team {team + 1}" if team >= 0 else "Unassigned"
            cv2.putText(annotated, team_name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw balls
        for ball in detections['balls']:
            x1, y1, x2, y2 = map(int, ball['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated, "BALL", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add stats overlay
        team_counts = defaultdict(int)
        for team in self.team_assignments.values():
            team_counts[team] += 1
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        cv2.putText(annotated, f"FPS: {self.fps:.1f} | SageMaker Powered", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Team 1: {team_counts[0]} players", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(annotated, f"Team 2: {team_counts[1]} players", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    def process_video(self, video_path, output_path='output_sagemaker.mp4'):
        """Process entire video using SageMaker"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        print(f"Processing video with SageMaker endpoint: {self.endpoint_name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = self.process_frame(frame, frame_count)
            out.write(annotated)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames: {frame_count}")
    
    def process_frames_directory(self, frames_dir, output_dir='backend/computerVision/annotated_frames', fps=30):
        """
        Process frames from a directory and save annotated frames back to disk.
        
        Args:
            frames_dir: Directory containing input frame images.
            output_dir: Directory to store annotated frames.
            fps: Ignored (kept for compatibility).
        """
        from pathlib import Path

        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob('*.jpg')) + sorted(frames_path.glob('*.png'))

        if not frame_files:
            print(f"ERROR: No frames found in {frames_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"Found {len(frame_files)} frames in {frames_dir}")
        print(f"Saving annotated frames to: {output_dir}")

        for frame_count, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"WARNING: Could not read {frame_file}, skipping...")
                continue

            annotated = self.process_frame(frame, frame_count)

            # Construct output path
            out_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(out_path, annotated)

            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{len(frame_files)} frames...")

        print(f"\n=== Frame Processing Complete ===")
        print(f"Total frames processed: {len(frame_files)}")
        print(f"Annotated frames saved to: {output_dir}")


def main():
    """Main function for SageMaker integration"""
    
    # ===== CONFIGURE YOUR SAGEMAKER ENDPOINT =====
    ENDPOINT_NAME = 'jumpstart-dft-mx-od-yolo3-mobilenet-20251025-183254'  # Replace with your endpoint
    AWS_REGION = "us-west-2"  # Replace with your region
    
    # Initialize tracker
    tracker = BasketballTrackerSageMaker(
        endpoint_name=ENDPOINT_NAME,
        region=AWS_REGION,
        confidence=0.3
    )
    
    # Process video
    video_path = "tidal_demo.mp4"
    tracker.process_frames_directory(
        frames_dir=r"backend\computerVision\frames",
        output_dir=r"backend\computerVision\annotated_frames",
        fps=30)


if __name__ == "__main__":
    main()