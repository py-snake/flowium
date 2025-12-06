"""
Simple Object Tracker using IoU (Intersection over Union)
Tracks vehicles across frames and assigns persistent IDs
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class VehicleTracker:
    """Track vehicles across frames using IoU matching"""

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        """
        Args:
            iou_threshold: Minimum IoU to consider a match
            max_age: Maximum frames to keep a track without detections
            min_hits: Minimum detections before confirming a track
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks = {}  # track_id -> track_data
        self.next_id = 1
        self.frame_count = 0

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_centroid_distance(self, box1, box2):
        """Calculate distance between box centroids"""
        # Calculate centroids
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2

        # Euclidean distance
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        return distance

    def calculate_box_size(self, box):
        """Calculate diagonal size of box for normalization"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        return np.sqrt(width**2 + height**2)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections using Hungarian-style greedy matching

        Args:
            detections: List of detection dicts with 'bbox', 'class_id', 'confidence', etc.

        Returns:
            List of tracked detections with 'track_id' added
        """
        self.frame_count += 1

        # Build cost matrix for all detection-track pairs
        match_matrix = []  # List of (score, det_idx, track_id)

        for det_idx, detection in enumerate(detections):
            bbox = detection['bbox']

            for track_id, track_data in self.tracks.items():
                # Skip if class doesn't match
                if detection['class_id'] != track_data['class_id']:
                    continue

                # Calculate IoU
                iou = self.calculate_iou(bbox, track_data['bbox'])

                # Calculate normalized centroid distance
                centroid_dist = self.calculate_centroid_distance(bbox, track_data['bbox'])
                box_size = self.calculate_box_size(track_data['bbox'])

                # Normalize distance by box size
                # Increased threshold to 4.0 for better tracking of fast-moving vehicles
                normalized_dist = centroid_dist / (box_size + 1e-6)

                # Hybrid matching score - use BOTH IoU and distance
                match_score = 0

                # Strategy: Combine IoU and distance scores
                if iou >= self.iou_threshold:
                    # Strong IoU match
                    match_score = iou + 0.5  # Boost IoU matches

                if normalized_dist < 4.0:
                    # Within proximity threshold
                    distance_score = max(0, 1.0 - (normalized_dist / 4.0))
                    # Take maximum of IoU-based and distance-based scores
                    match_score = max(match_score, distance_score)

                # Only consider matches with non-zero score
                if match_score > 0:
                    match_matrix.append((match_score, det_idx, track_id))

        # Sort by score (highest first) for greedy assignment
        match_matrix.sort(reverse=True, key=lambda x: x[0])

        # Greedy assignment: assign best matches first
        matched_detections = set()
        matched_tracks = set()
        detection_to_track = {}

        for score, det_idx, track_id in match_matrix:
            # Skip if already matched
            if det_idx in matched_detections or track_id in matched_tracks:
                continue

            # Make the match
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)
            detection_to_track[det_idx] = track_id

        # Update tracks and create tracked detections list
        tracked_detections = []

        for det_idx, detection in enumerate(detections):
            bbox = detection['bbox']

            if det_idx in detection_to_track:
                # Matched to existing track
                track_id = detection_to_track[det_idx]

                # Update existing track
                self.tracks[track_id].update({
                    'bbox': bbox,
                    'last_seen': self.frame_count,
                    'hits': self.tracks[track_id]['hits'] + 1,
                    'confidence': detection['confidence']
                })
            else:
                # Create new track for unmatched detection
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'hits': 1,
                    'confidence': detection['confidence'],
                    'stored': False  # Flag to track if we've stored this in DB
                }
                matched_tracks.add(track_id)

            # Add track_id to detection
            detection_copy = detection.copy()
            detection_copy['track_id'] = track_id
            detection_copy['hits'] = self.tracks[track_id]['hits']
            tracked_detections.append(detection_copy)

        # Remove old tracks that haven't been seen
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if self.frame_count - track_data['last_seen'] > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return tracked_detections

    def get_confirmed_new_tracks(self, tracked_detections: List[Dict]) -> List[Dict]:
        """
        Get only NEW confirmed tracks (that haven't been stored yet)

        A track is confirmed if it has been detected in min_hits frames
        Only returns tracks that haven't been stored in DB yet

        Returns:
            List of detections for new confirmed tracks
        """
        new_tracks = []

        for detection in tracked_detections:
            track_id = detection['track_id']
            track = self.tracks[track_id]

            # Check if track is confirmed and not yet stored
            if track['hits'] >= self.min_hits and not track['stored']:
                new_tracks.append(detection)
                # Mark as stored
                track['stored'] = True

        return new_tracks

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks"""
        return len(self.tracks)

    def get_confirmed_track_count(self) -> int:
        """Get number of confirmed tracks"""
        return sum(1 for t in self.tracks.values() if t['hits'] >= self.min_hits)
