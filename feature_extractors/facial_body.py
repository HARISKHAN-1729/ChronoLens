"""
Facial/Body Language feature extractor
Uses MediaPipe for face/pose detection and expression recognition
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class FacialBodyExtractor(FeatureExtractor):
    """Extract facial expressions and body language features"""
    
    def extract(self):
        """Extract all facial and body language features"""
        faces_detected = []
        expressions = []
        head_movements = []
        body_poses = []
        
        prev_face_landmarks = None
        prev_pose_landmarks = None
        
        # Get MediaPipe models
        face_mesh = self.model_loader.get_mediapipe_face_mesh()
        pose = self.model_loader.get_mediapipe_pose()
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=2):
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    faces_detected.append({
                        "timestamp": timestamp,
                        "frame": frame_num
                    })
                    
                    # Analyze expression
                    expression = self._analyze_expression(face_landmarks, frame.shape)
                    expressions.append({
                        "timestamp": timestamp,
                        "expression": expression["type"],
                        "confidence": expression["confidence"]
                    })
                    
                    # Detect head movement
                    if prev_face_landmarks:
                        movement = self._detect_head_movement(face_landmarks, prev_face_landmarks)
                        if movement:
                            head_movements.append({
                                "timestamp": timestamp,
                                "movement_type": movement["type"],
                                "magnitude": movement["magnitude"]
                            })
                    
                    prev_face_landmarks = face_landmarks
            
            # Detect body pose
            pose_results = pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                body_poses.append({
                    "timestamp": timestamp,
                    "frame": frame_num
                })
                
                # Analyze posture
                posture = self._analyze_posture(pose_results.pose_landmarks)
                
                # Detect gestures
                if prev_pose_landmarks:
                    gesture = self._detect_gesture(pose_results.pose_landmarks, prev_pose_landmarks)
                
                prev_pose_landmarks = pose_results.pose_landmarks
        
        # Calculate aggregate metrics
        expression_distribution = self._get_expression_distribution(expressions)
        engagement_score = self._calculate_engagement(faces_detected, expressions)
        
        return {
            "total_faces_detected": len(faces_detected),
            "face_detection_rate": float(len(faces_detected) / self.total_frames) if self.total_frames > 0 else 0.0,
            "total_expressions_detected": len(expressions),
            "expression_distribution": expression_distribution,
            "total_head_movements": len(head_movements),
            "head_movement_rate": float(len(head_movements) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "total_body_poses_detected": len(body_poses),
            "engagement_score": float(engagement_score),
            "average_expression_confidence": float(np.mean([e["confidence"] for e in expressions])) if expressions else 0.0,
            "detailed_expressions": expressions[:50],
            "detailed_head_movements": head_movements[:50],
            "expression_timeline": [{"timestamp": e["timestamp"], "expression": e["expression"]} for e in expressions[:50]]
        }
    
    def _analyze_expression(self, face_landmarks, frame_shape):
        """Analyze facial expression from landmarks"""
        # Extract key points for expression analysis
        landmarks = face_landmarks.landmark
        h, w = frame_shape[:2]
        
        # Mouth aspect ratio (smile detection)
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_left = landmarks[61].x
        mouth_right = landmarks[291].x
        
        mouth_height = abs(mouth_bottom - mouth_top)
        mouth_width = abs(mouth_right - mouth_left)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Eye aspect ratio (surprise/focus)
        left_eye_top = landmarks[159].y
        left_eye_bottom = landmarks[145].y
        eye_height = abs(left_eye_top - left_eye_bottom)
        
        # Eyebrow position (surprise/concern)
        eyebrow_y = landmarks[70].y
        
        # Classify expression
        if mouth_ratio > 0.3:
            return {"type": "surprised", "confidence": 0.7}
        elif mouth_width > 0.15:
            return {"type": "smiling", "confidence": 0.8}
        elif eyebrow_y < 0.3:
            return {"type": "concerned", "confidence": 0.6}
        elif eye_height < 0.02:
            return {"type": "focused", "confidence": 0.7}
        else:
            return {"type": "neutral", "confidence": 0.5}
    
    def _detect_head_movement(self, current_landmarks, prev_landmarks):
        """Detect head movement between frames"""
        # Compare nose tip position
        curr_nose = current_landmarks.landmark[1]
        prev_nose = prev_landmarks.landmark[1]
        
        dx = curr_nose.x - prev_nose.x
        dy = curr_nose.y - prev_nose.y
        
        magnitude = np.sqrt(dx**2 + dy**2)
        
        if magnitude > 0.02:
            if abs(dx) > abs(dy):
                movement_type = "turn_left" if dx < 0 else "turn_right"
            else:
                movement_type = "tilt_up" if dy < 0 else "tilt_down"
            
            return {
                "type": movement_type,
                "magnitude": float(magnitude)
            }
        
        return None
    
    def _analyze_posture(self, pose_landmarks):
        """Analyze body posture"""
        landmarks = pose_landmarks.landmark
        
        # Shoulder alignment
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        shoulder_slope = (right_shoulder.y - left_shoulder.y) / (right_shoulder.x - left_shoulder.x) if right_shoulder.x != left_shoulder.x else 0
        
        # Spine alignment
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Classify posture
        if abs(shoulder_slope) < 0.1:
            return "upright"
        elif shoulder_slope > 0.1:
            return "leaning_right"
        else:
            return "leaning_left"
    
    def _detect_gesture(self, current_pose, prev_pose):
        """Detect hand gestures"""
        curr_landmarks = current_pose.landmark
        prev_landmarks = prev_pose.landmark
        
        # Track right hand movement
        curr_hand = curr_landmarks[16]  # Right wrist
        prev_hand = prev_landmarks[16]
        
        dx = curr_hand.x - prev_hand.x
        dy = curr_hand.y - prev_hand.y
        
        magnitude = np.sqrt(dx**2 + dy**2)
        
        if magnitude > 0.05:
            if abs(dy) > abs(dx) and dy < 0:
                return "pointing"
            elif abs(dx) > abs(dy):
                return "waving"
        
        return None
    
    def _get_expression_distribution(self, expressions):
        """Get distribution of expressions"""
        distribution = defaultdict(int)
        
        for expr in expressions:
            distribution[expr["expression"]] += 1
        
        total = len(expressions)
        
        return {k: v / total if total > 0 else 0.0 for k, v in distribution.items()}
    
    def _calculate_engagement(self, faces_detected, expressions):
        """Calculate engagement score based on face presence and expressions"""
        if not faces_detected:
            return 0.0
        
        # Face presence score
        presence_score = len(faces_detected) / self.total_frames if self.total_frames > 0 else 0
        
        # Expression variety score
        unique_expressions = len(set(e["expression"] for e in expressions))
        variety_score = min(unique_expressions / 5, 1.0)  # Normalize to max 5 expressions
        
        # Combine scores
        engagement = (presence_score * 0.6 + variety_score * 0.4)
        
        return engagement
