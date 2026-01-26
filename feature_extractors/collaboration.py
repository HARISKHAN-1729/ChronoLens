"""
Collaboration feature extractor
Detects multiple users, turn-taking, and communication patterns
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class CollaborationExtractor(FeatureExtractor):
    """Extract collaboration and multi-user interaction features"""
    
    def extract(self):
        """Extract all collaboration features"""
        users_detected = []
        turn_taking_events = []
        communication_events = []
        concurrent_actions = []
        
        prev_frame = None
        current_user = None
        user_history = []
        
        # Get models
        face_mesh = self.model_loader.get_mediapipe_face_mesh()
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=3):
            # Detect faces (multiple users)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)
            
            num_faces = 0
            if face_results.multi_face_landmarks:
                num_faces = len(face_results.multi_face_landmarks)
                
                users_detected.append({
                    "timestamp": timestamp,
                    "num_users": num_faces,
                    "frame": frame_num
                })
                
                # Detect turn-taking (change in active user)
                if current_user != num_faces:
                    turn_taking_events.append({
                        "timestamp": timestamp,
                        "from_users": current_user,
                        "to_users": num_faces
                    })
                    current_user = num_faces
            
            if prev_frame is not None:
                # Detect concurrent actions (multiple interaction points)
                concurrent = self._detect_concurrent_actions(frame, prev_frame)
                if concurrent:
                    concurrent_actions.append({
                        "timestamp": timestamp,
                        "num_actions": concurrent
                    })
                
                # Detect communication (audio would be better, but use visual cues)
                communication = self._detect_communication(frame, face_results)
                if communication:
                    communication_events.append({
                        "timestamp": timestamp,
                        "type": communication
                    })
            
            prev_frame = frame.copy()
            user_history.append(num_faces)
            
            # Keep recent history
            if len(user_history) > 100:
                user_history.pop(0)
        
        # Calculate collaboration metrics
        max_users = max(user_history) if user_history else 0
        avg_users = np.mean(user_history) if user_history else 0
        collaboration_detected = max_users > 1
        
        return {
            "collaboration_detected": collaboration_detected,
            "max_concurrent_users": int(max_users),
            "average_users": float(avg_users),
            "total_turn_taking_events": len(turn_taking_events),
            "turn_taking_rate": float(len(turn_taking_events) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "total_communication_events": len(communication_events),
            "total_concurrent_actions": len(concurrent_actions),
            "collaboration_intensity": self._calculate_collaboration_intensity(users_detected, turn_taking_events),
            "user_distribution": self._get_user_distribution(user_history),
            "detailed_turn_taking": turn_taking_events[:50],
            "detailed_communication": communication_events[:50],
            "detailed_concurrent_actions": concurrent_actions[:50],
            "user_timeline": users_detected[:100]
        }
    
    def _detect_concurrent_actions(self, frame, prev_frame):
        """Detect multiple simultaneous interaction points"""
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant change regions
        significant_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                significant_regions += 1
        
        # Multiple regions suggest concurrent actions
        if significant_regions >= 2:
            return significant_regions
        
        return None
    
    def _detect_communication(self, frame, face_results):
        """Detect communication between users"""
        if not face_results or not face_results.multi_face_landmarks:
            return None
        
        # Check for mouth movements (speaking)
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Mouth aspect ratio
            mouth_top = landmarks[13].y
            mouth_bottom = landmarks[14].y
            mouth_height = abs(mouth_bottom - mouth_top)
            
            if mouth_height > 0.03:  # Mouth open (speaking)
                return "verbal_communication"
        
        # Check for pointing gestures (would need pose detection)
        return None
    
    def _calculate_collaboration_intensity(self, users_detected, turn_taking_events):
        """Calculate intensity of collaboration (0-1)"""
        if not users_detected:
            return 0.0
        
        # Factor 1: Multiple users present
        multi_user_frames = sum(1 for u in users_detected if u["num_users"] > 1)
        user_factor = multi_user_frames / len(users_detected) if users_detected else 0
        
        # Factor 2: Turn-taking frequency
        turn_factor = min(len(turn_taking_events) / 20, 1.0)  # Normalize to 20 turns
        
        # Combine factors
        intensity = (user_factor * 0.6 + turn_factor * 0.4)
        
        return float(intensity)
    
    def _get_user_distribution(self, user_history):
        """Get distribution of user counts"""
        distribution = defaultdict(int)
        
        for count in user_history:
            distribution[count] += 1
        
        total = len(user_history)
        
        return {
            str(k): v / total if total > 0 else 0.0 
            for k, v in distribution.items()
        }
