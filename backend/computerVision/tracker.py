import cv2
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None

import numpy as np
from collections import defaultdict
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import os

from ultralytics import YOLO
import supervision as sv

class BasketballTrackerLive:
    def __init__(self, model_path='yolov8x.pt', confidence=0.3):
        """
        Initialize the basketball tracker for live feed

        Args:
            model_path: Path to YOLO model (yolov8x.pt has person, sports ball classes)
            confidence: Detection confidence threshold
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence = confidence

        self.tracker = sv.ByteTrack()

        self.team_assignments = {}
        self.player_embeddings = defaultdict(list)

        # Team colors (BGR format for OpenCV)
        self.team_colors = {
            0: (255, 0, 0),    # Blue for Team 1
            1: (0, 0, 255),    # Red for Team 2
            -1: (128, 128, 128) # Gray for unassigned
        }

        # FPS tracking
        self.fps = 0
        self.frame_times = []

        self.hoop_bbox = None            # (x1, y1, x2, y2) - hoop region
        self.team_shots = {0: 0, 1: 0}  # successful shots per team
        self.team_attempts = {0: 0, 1: 0}  # total shot attempts per team
        self.ball_possession = None      # track_id of player with ball, or None
        self.shot_in_progress = False    # flag for ongoing shot attempt
        self.last_shot_attempt_time = 0  # timestamp of last shot attempt
        self.shot_cooldown = 1.0         # seconds cooldown after shot
        self.shot_team = -1              # team that attempted last shot

        self.last_possession_team = -1   # -1 means no last possession
        self.shot_buffer_frames = 40     # Number of frames to ignore after a shot
        self.shot_buffer_counter = 0     # Counts frames since last shot

    def set_hoop_region(self, frame_width, frame_height):
        """
        Set the hoop region to a fixed top-center area (fallback)

        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        # Define hoop as top-center region (adjust these percentages as needed)
        hoop_width = int(frame_width * 0.2)  # 20% of frame width
        hoop_height = int(frame_height * 0.15)  # 15% of frame height
        hoop_x1 = (frame_width - hoop_width) // 2
        hoop_y1 = 0
        hoop_x2 = hoop_x1 + hoop_width
        hoop_y2 = hoop_height

        self.hoop_bbox = (hoop_x1, hoop_y1, hoop_x2, hoop_y2)
        print(f"Hoop region set to default: {self.hoop_bbox}")

    def check_bbox_overlap(self, bbox1, bbox2):
        """
        Check if two bounding boxes overlap

        Args:
            bbox1, bbox2: (x1, y1, x2, y2)

        Returns:
            True if boxes overlap, False otherwise
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Check for overlap
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def is_point_in_bbox(self, point, bbox):
        """
        Check if a point is inside a bounding box

        Args:
            point: (x, y)
            bbox: (x1, y1, x2, y2)

        Returns:
            True if point is inside bbox, False otherwise
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
        
    def extract_color_histogram(self, image, bbox):
        """
        Extract color histogram from player crop as a simple embedding
        
        Args:
            image: Full frame
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Flattened color histogram
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid coordinates
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return np.zeros(48)  # Return zero vector for invalid crops
        
        # Take center crop to avoid background
        h_crop, w_crop = crop.shape[:2]
        center_h = int(h_crop * 0.3), int(h_crop * 0.7)
        center_w = int(w_crop * 0.3), int(w_crop * 0.7)
        center_crop = crop[center_h[0]:center_h[1], center_w[0]:center_w[1]]
        
        if center_crop.size == 0:
            center_crop = crop
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for H, S, V channels
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize and flatten
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate all histograms
        embedding = np.concatenate([hist_h, hist_s, hist_v])
        
        return embedding
    
    def cluster_teams(self, embeddings_dict, min_samples=5, balanced=True):
        """
        Cluster players into teams based on their visual embeddings
        
        Args:
            embeddings_dict: Dictionary mapping track_id to list of embeddings
            min_samples: Minimum number of embeddings needed per player
            balanced: If True, enforce equal team sizes (reject odd number of players)
        
        Returns:
            Dictionary mapping track_id to team_label (0 or 1)
        """
        # Filter players with enough samples
        valid_ids = [tid for tid, embs in embeddings_dict.items() 
                     if len(embs) >= min_samples]
        
        # Handle edge cases
        if len(valid_ids) < 2:
            if len(valid_ids) == 1:
                return {valid_ids[0]: 0}  # Assign single player to team 0
            return {}
        
        # Check if balanced teams are possible
        if balanced and len(valid_ids) % 2 != 0:
            # Odd number of players - can't balance teams
            # Return empty to wait for even number
            return {}
        
        # Average embeddings for each player
        avg_embeddings = []
        for tid in valid_ids:
            embs = np.array(embeddings_dict[tid])
            avg_emb = np.mean(embs, axis=0)
            avg_embeddings.append(avg_emb)
        
        avg_embeddings = np.array(avg_embeddings)
        
        # Optional: dimensionality reduction if needed
        n_samples = avg_embeddings.shape[0]
        n_features = avg_embeddings.shape[1]
        
        if n_features > 10 and n_samples > 10:
            pca = PCA(n_components=10)
            avg_embeddings = pca.fit_transform(avg_embeddings)
        elif n_features > n_samples:
            pca = PCA(n_components=min(n_samples - 1, n_features))
            avg_embeddings = pca.fit_transform(avg_embeddings)
        
        # K-means clustering (k=2 for two teams)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(avg_embeddings)
        
        # If balanced mode, verify equal distribution
        if balanced:
            team_0_count = np.sum(team_labels == 0)
            team_1_count = np.sum(team_labels == 1)
            
            # If imbalanced, rebalance by reassigning furthest players
            if team_0_count != team_1_count:
                # Calculate distances to cluster centers
                distances = kmeans.transform(avg_embeddings)
                
                # Determine which team has more players
                if team_0_count > team_1_count:
                    excess_team = 0
                    deficit_team = 1
                else:
                    excess_team = 1
                    deficit_team = 0
                
                # Find players to reassign (those closest to the other cluster)
                excess_mask = team_labels == excess_team
                excess_indices = np.where(excess_mask)[0]
                
                # Get distances from excess team players to deficit team center
                distances_to_deficit = distances[excess_mask, deficit_team]
                
                # How many to reassign?
                num_to_reassign = abs(team_0_count - team_1_count) // 2
                
                # Reassign closest players to balance
                closest_indices = np.argsort(distances_to_deficit)[:num_to_reassign]
                for idx in closest_indices:
                    team_labels[excess_indices[idx]] = deficit_team
        
        # Create mapping
        team_assignment = {}
        for tid, label in zip(valid_ids, team_labels):
            team_assignment[tid] = int(label)
        
        return team_assignment
    
    def update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
    
    def process_frame(self, frame, frame_count, cluster_interval=60):
        # Update FPS
        self.update_fps()

        # Set hoop region if not set
        if self.hoop_bbox is None:
            h, w = frame.shape[:2]
            self.set_hoop_region(w, h)

        # Handle shot buffer period
        if self.shot_buffer_counter > 0:
            self.shot_buffer_counter -= 1
            annotated = frame.copy()
            cv2.putText(annotated, f"SHOT BUFFER - PAUSED ({self.shot_buffer_counter})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            return annotated, self.last_possession_team

        # Check if in cooldown period
        current_time = time.time()
        in_cooldown = (current_time - self.last_shot_attempt_time) < self.shot_cooldown

        if in_cooldown:
            annotated = frame.copy()
            cv2.putText(annotated, "SHOT COOLDOWN - PAUSED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            return annotated, self.last_possession_team

        # Run detection
        results = self.model(frame, conf=self.confidence, classes=[0, 32], verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)
        player_mask = results.boxes.cls.cpu().numpy() == 0
        ball_mask = results.boxes.cls.cpu().numpy() == 32

        player_detections = detections[player_mask] if player_mask.any() else sv.Detections.empty()
        ball_detections = detections[ball_mask] if ball_mask.any() else sv.Detections.empty()

        # Track players
        if len(player_detections) > 0:
            player_detections = self.tracker.update_with_detections(player_detections)
            if not self.shot_in_progress:
                for i, (bbox, track_id) in enumerate(zip(player_detections.xyxy,
                                                        player_detections.tracker_id)):
                    embedding = self.extract_color_histogram(frame, bbox)
                    self.player_embeddings[track_id].append(embedding)
                    if len(self.player_embeddings[track_id]) > 50:
                        self.player_embeddings[track_id].pop(0)

        # Periodically re-cluster teams
        if frame_count % cluster_interval == 0 and len(self.player_embeddings) > 0 and not self.shot_in_progress:
            self.team_assignments = self.cluster_teams(self.player_embeddings)

        # Ball possession and shot tracking
        ball_overlaps_hoop = False
        if len(ball_detections) > 0 and len(player_detections) > 0:
            ball_bbox = ball_detections.xyxy[0]

            if self.hoop_bbox:
                ball_overlaps_hoop = self.check_bbox_overlap(ball_bbox, self.hoop_bbox)

            current_possession = None
            for player_bbox, track_id in zip(player_detections.xyxy, player_detections.tracker_id):
                if self.check_bbox_overlap(ball_bbox, player_bbox):
                    current_possession = track_id
                    break

            # Detect shot attempt: ball was possessed but now not
            if self.ball_possession is not None and current_possession is None and not self.shot_in_progress:
                self.shot_in_progress = True
                self.shot_team = self.team_assignments.get(self.ball_possession, -1)
                if self.shot_team >= 0:
                    self.team_attempts[self.shot_team] += 1
                self.last_shot_attempt_time = current_time
                print(f"Shot attempt detected by Team {self.shot_team + 1}")

            # Detect shot success: ball overlaps hoop during shot attempt
            if self.shot_in_progress and self.hoop_bbox and ball_overlaps_hoop:
                if self.shot_team >= 0:
                    self.team_shots[self.shot_team] += 1
                    print(f"SHOT MADE by Team {self.shot_team + 1}! Total: {self.team_shots[self.shot_team]}")
                self.shot_in_progress = False
                self.shot_buffer_counter = self.shot_buffer_frames  # Start buffer
                self.last_shot_attempt_time = current_time

            self.ball_possession = current_possession

        # Update last possession team
        if self.ball_possession is not None:
            self.last_possession_team = self.team_assignments.get(self.ball_possession, -1)

        # Annotate frame (rest of your existing code...)
        annotated = frame.copy()
        # ...drawing code stays the same...

        return annotated, self.last_possession_team

    
    def run_webcam(self, camera_id=0, cluster_interval=60):
        """
        Run tracker on webcam feed

        Args:
            camera_id: Camera device ID (0 for default webcam)
            cluster_interval: How often to re-cluster teams (in frames) - increased for latency
        """
        print(f"Opening webcam (ID: {camera_id})...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Webcam opened successfully!")
        print("Press 'q' to quit, 'r' to reset teams, 'h' to re-detect rim")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            annotated = self.process_frame(frame, frame_count, cluster_interval)

            # Display
            cv2.imshow('Basketball Tracker - LIVE', annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Resetting team assignments...")
                self.team_assignments = {}
                self.player_embeddings = defaultdict(list)
                self.team_shots = {0: 0, 1: 0}  # Reset shots too
            elif key == ord('h'):
                print("Resetting hoop region...")
                self.hoop_bbox = None  # Will be reset on next frame

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")
    
    def run_stream(self, stream_url, cluster_interval=60, save_output=False,
                   output_path='output_stream.mp4'):
        """
        Run tracker on video stream (RTSP, HTTP, etc.)

        Args:
            stream_url: URL of the video stream
            cluster_interval: How often to re-cluster teams (in frames) - increased for latency
            save_output: Whether to save output video
            output_path: Path to save output if save_output=True
        """
        print(f"Connecting to stream: {stream_url}")
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print("Error: Could not open stream")
            return

        print("Stream connected successfully!")
        print("Press 'q' to quit, 'r' to reset teams, 'h' to re-detect rim")

        # Setup video writer if saving
        out = None
        if save_output:
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from stream")
                break

            # Process frame
            annotated = self.process_frame(frame, frame_count, cluster_interval)

            # Save frame if recording
            if save_output and out is not None:
                out.write(annotated)

            # Display
            cv2.imshow('Basketball Tracker - STREAM', annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Resetting team assignments...")
                self.team_assignments = {}
                self.player_embeddings = defaultdict(list)
                self.team_shots = {0: 0, 1: 0}  # Reset shots too
            elif key == ord('h'):
                print("Resetting hoop region...")
                self.hoop_bbox = None  # Will be reset on next frame

            frame_count += 1

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Stream closed")


def main():
    """
    Main function - choose your video source
    """
    # Initialize tracker
    tracker = BasketballTrackerLive(
        model_path='backend/computerVision/yolov8n.pt',  # nano for low latency
        confidence=0.1,
        #rim_model_path='backend/computerVision/rim_model.pt'  # Path to rim detection model (optional)
    )
    
    # ===== CHOOSE ONE OF THE OPTIONS BELOW =====
    
    # Option 1: Use webcam
    tracker.run_webcam(camera_id=0, cluster_interval=30)
    
    # Option 2: Use RTSP stream (IP camera, etc.)
    # stream_url = "rtsp://username:password@ip_address:port/stream"
    # tracker.run_stream(stream_url, cluster_interval=30, save_output=False)
    
    # Option 3: Use HTTP stream
    # stream_url = "http://your-stream-url.com/stream.mjpg"
    # tracker.run_stream(stream_url, cluster_interval=30, save_output=True)
    
    # Option 4: Use YouTube live stream (requires youtube-dl/yt-dlp)
    # stream_url = "https://www.youtube.com/watch?v=VIDEO_ID"
    # tracker.run_stream(stream_url, cluster_interval=30, save_output=False)


if __name__ == "__main__":
    main()