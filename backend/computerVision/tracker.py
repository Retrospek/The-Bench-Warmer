import cv2
import numpy as np
from collections import defaultdict
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===== INSTALLATION REQUIRED =====
# pip install opencv-python numpy torch torchvision ultralytics scikit-learn pillow supervision

from ultralytics import YOLO
import supervision as sv

class BasketballTracker:
    def __init__(self, model_path='yolov8x.pt', confidence=0.3):
        """
        Initialize the basketball tracker
        
        Args:
            model_path: Path to YOLO model (yolov8x.pt has person, sports ball classes)
            confidence: Detection confidence threshold
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
        
        # Team assignment storage
        self.team_assignments = {}  # track_id -> team_label
        self.player_embeddings = defaultdict(list)  # track_id -> list of embeddings
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
        
        # Team colors (BGR format for OpenCV)
        self.team_colors = {
            0: (255, 0, 0),    # Blue for Team 1
            1: (0, 0, 255),    # Red for Team 2
            -1: (128, 128, 128) # Gray for unassigned
        }
        
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
    
    def cluster_teams(self, embeddings_dict, min_samples=5):
        """
        Cluster players into teams based on their visual embeddings
        
        Args:
            embeddings_dict: Dictionary mapping track_id to list of embeddings
            min_samples: Minimum number of embeddings needed per player
        
        Returns:
            Dictionary mapping track_id to team_label (0 or 1)
        """
        # Filter players with enough samples
        valid_ids = [tid for tid, embs in embeddings_dict.items() 
                     if len(embs) >= min_samples]
        
        if len(valid_ids) < 2:
            return {}
        
        # Average embeddings for each player
        avg_embeddings = []
        for tid in valid_ids:
            embs = np.array(embeddings_dict[tid])
            avg_emb = np.mean(embs, axis=0)
            avg_embeddings.append(avg_emb)
        
        avg_embeddings = np.array(avg_embeddings)
        
        # Optional: dimensionality reduction if needed
        if avg_embeddings.shape[1] > 10:
            pca = PCA(n_components=10)
            avg_embeddings = pca.fit_transform(avg_embeddings)
        
        # K-means clustering (k=2 for two teams)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        team_labels = kmeans.fit_predict(avg_embeddings)
        
        # Create mapping
        team_assignment = {}
        for tid, label in zip(valid_ids, team_labels):
            team_assignment[tid] = int(label)
        
        return team_assignment
    
    def process_frame(self, frame, frame_count, cluster_interval=30):
        """
        Process a single frame: detect, track, and assign teams
        
        Args:
            frame: Input frame
            frame_count: Current frame number
            cluster_interval: How often to re-cluster teams (in frames)
        
        Returns:
            Annotated frame
        """
        # Run detection
        results = self.model(frame, conf=self.confidence, classes=[0, 32])[0]
        # classes=[0, 32] -> 0: person, 32: sports ball
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Separate players and balls
        player_mask = results.boxes.cls.cpu().numpy() == 0  # person class
        ball_mask = results.boxes.cls.cpu().numpy() == 32   # sports ball class
        
        player_detections = detections[player_mask] if player_mask.any() else sv.Detections.empty()
        ball_detections = detections[ball_mask] if ball_mask.any() else sv.Detections.empty()
        
        # Track players
        if len(player_detections) > 0:
            player_detections = self.tracker.update_with_detections(player_detections)
            
            # Extract embeddings for team clustering
            for i, (bbox, track_id) in enumerate(zip(player_detections.xyxy, 
                                                      player_detections.tracker_id)):
                embedding = self.extract_color_histogram(frame, bbox)
                self.player_embeddings[track_id].append(embedding)
        
        # Periodically re-cluster teams
        if frame_count % cluster_interval == 0 and len(self.player_embeddings) > 0:
            print(f"Frame {frame_count}: Clustering teams...")
            self.team_assignments = self.cluster_teams(self.player_embeddings)
            print(f"Assigned {len(self.team_assignments)} players to teams")
        
        # Annotate frame
        annotated = frame.copy()
        
        # Draw player boxes with team colors
        if len(player_detections) > 0:
            for i, (bbox, track_id) in enumerate(zip(player_detections.xyxy, 
                                                      player_detections.tracker_id)):
                team = self.team_assignments.get(track_id, -1)
                color = self.team_colors[team]
                
                # Draw box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                team_name = f"Team {team + 1}" if team >= 0 else "Unassigned"
                label = f"{team_name} (ID: {track_id})"
                
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball
        if len(ball_detections) > 0:
            for bbox in ball_detections.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(annotated, "BALL", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add stats
        team_counts = defaultdict(int)
        for team in self.team_assignments.values():
            team_counts[team] += 1
        
        stats_y = 30
        cv2.putText(annotated, f"Frame: {frame_count}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Team 1: {team_counts[0]} | Team 2: {team_counts[1]}", 
                   (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def process_video(self, video_path, output_path='output_basketball.mp4', 
                     max_frames=None):
        """
        Process entire video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            max_frames: Maximum frames to process (None for all)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Process frame
            annotated = self.process_frame(frame, frame_count)
            
            # Write frame
            out.write(annotated)
            
            # Display (optional)
            cv2.imshow('Basketball Tracker', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")

