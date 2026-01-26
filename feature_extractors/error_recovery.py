"""
Error Recovery feature extractor
Detects user errors, backtracking, and recovery patterns
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import ERROR_CONFIG

class ErrorRecoveryExtractor(FeatureExtractor):
    """Extract error and recovery behavior features"""
    
    def extract(self):
        """Extract all error recovery features"""
        errors = []
        recovery_events = []
        backtrack_events = []
        repeated_errors = []
        
        prev_frame = None
        action_history = []
        location_history = []
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=1):
            if prev_frame is not None:
                # Detect error indicators
                error_detected, error_type = self._detect_error(frame, prev_frame)
                
                if error_detected:
                    errors.append({
                        "timestamp": timestamp,
                        "frame": frame_num,
                        "type": error_type,
                        "recovery_start": timestamp
                    })
                
                # Detect backtracking (returning to previous locations/actions)
                backtrack = self._detect_backtrack(frame, location_history)
                if backtrack:
                    backtrack_events.append({
                        "timestamp": timestamp,
                        "frame": frame_num,
                        "previous_location": backtrack
                    })
                
                # Track action locations for pattern detection
                action_loc = self._get_action_location(frame, prev_frame)
                if action_loc:
                    location_history.append({
                        "timestamp": timestamp,
                        "location": action_loc
                    })
                    
                    # Keep only recent history
                    if len(location_history) > 50:
                        location_history.pop(0)
                
                # Check for recovery completion
                if errors and not errors[-1].get("recovery_end"):
                    if self._is_recovery_complete(frame, prev_frame):
                        errors[-1]["recovery_end"] = timestamp
                        errors[-1]["recovery_time"] = timestamp - errors[-1]["recovery_start"]
                        
                        recovery_events.append({
                            "error_timestamp": errors[-1]["timestamp"],
                            "recovery_time": errors[-1]["recovery_time"],
                            "error_type": errors[-1]["type"]
                        })
            
            prev_frame = frame.copy()
        
        # Detect repeated errors (same location/action within time window)
        repeated_errors = self._detect_repeated_errors(errors)
        
        # Calculate metrics
        total_errors = len(errors)
        avg_recovery_time = np.mean([e["recovery_time"] for e in errors if "recovery_time" in e]) if errors else 0.0
        error_rate = total_errors / (self.duration / 60) if self.duration > 0 else 0.0
        
        return {
            "total_errors": total_errors,
            "total_recovery_events": len(recovery_events),
            "total_backtrack_events": len(backtrack_events),
            "total_repeated_errors": len(repeated_errors),
            "average_recovery_time": float(avg_recovery_time),
            "error_rate_per_minute": float(error_rate),
            "error_types_distribution": self._get_error_distribution(errors),
            "successful_recoveries": sum(1 for e in errors if "recovery_time" in e),
            "recovery_success_rate": sum(1 for e in errors if "recovery_time" in e) / total_errors if total_errors > 0 else 0.0,
            "detailed_errors": errors[:50],
            "detailed_recovery_events": recovery_events[:50],
            "detailed_backtrack_events": backtrack_events[:50],
            "repeated_error_patterns": repeated_errors[:20]
        }
    
    def _detect_error(self, frame, prev_frame):
        """Detect error indicators in the video"""
        # Error detection strategies:
        # 1. Red/warning colors appearing
        # 2. Rapid clicking in same area (frustration)
        # 3. Sudden UI changes (error dialogs)
        
        # Check for red/warning colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_pixels = np.sum(red_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / total_pixels
        
        # Sudden appearance of red (error messages)
        if red_ratio > 0.01:
            hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
            mask1_prev = cv2.inRange(hsv_prev, lower_red1, upper_red1)
            mask2_prev = cv2.inRange(hsv_prev, lower_red2, upper_red2)
            red_mask_prev = mask1_prev + mask2_prev
            red_pixels_prev = np.sum(red_mask_prev > 0)
            
            if red_pixels > red_pixels_prev * 2:
                return True, "error_message"
        
        # Check for rapid changes (wrong clicks)
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > 50:
            return True, "wrong_click"
        
        return False, None
    
    def _detect_backtrack(self, frame, location_history):
        """Detect backtracking to previous locations"""
        if len(location_history) < 5:
            return None
        
        # Get current action location
        current_features = self._extract_frame_features(frame)
        
        # Check if similar to recent past locations
        for i in range(len(location_history) - 5, max(0, len(location_history) - 20), -1):
            past_loc = location_history[i]["location"]
            similarity = self._calculate_location_similarity(current_features, past_loc)
            
            if similarity > 0.8:
                return location_history[i]["timestamp"]
        
        return None
    
    def _get_action_location(self, frame, prev_frame):
        """Get location of current action"""
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        return None
    
    def _extract_frame_features(self, frame):
        """Extract features for location comparison"""
        # Simple feature: average color in grid cells
        h, w = frame.shape[:2]
        features = []
        
        for i in range(4):
            for j in range(4):
                cell = frame[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                avg_color = np.mean(cell, axis=(0, 1))
                features.extend(avg_color)
        
        return np.array(features)
    
    def _calculate_location_similarity(self, features1, location):
        """Calculate similarity between current and past location"""
        if location is None or not isinstance(location, tuple):
            return 0.0
        
        # Simple distance-based similarity
        # In real implementation, would compare visual features
        return 0.5  # Placeholder
    
    def _is_recovery_complete(self, frame, prev_frame):
        """Check if recovery from error is complete"""
        # Recovery indicated by stable state (low frame difference)
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        return mean_diff < 10
    
    def _detect_repeated_errors(self, errors):
        """Detect repeated errors in same location/type"""
        repeated = []
        window = ERROR_CONFIG["repeated_error_window"]
        
        for i in range(len(errors)):
            for j in range(i + 1, len(errors)):
                time_diff = errors[j]["timestamp"] - errors[i]["timestamp"]
                
                if time_diff > window:
                    break
                
                if errors[i]["type"] == errors[j]["type"]:
                    repeated.append({
                        "first_occurrence": errors[i]["timestamp"],
                        "second_occurrence": errors[j]["timestamp"],
                        "time_between": time_diff,
                        "error_type": errors[i]["type"]
                    })
        
        return repeated
    
    def _get_error_distribution(self, errors):
        """Get distribution of error types"""
        distribution = defaultdict(int)
        
        for error in errors:
            distribution[error["type"]] += 1
        
        return dict(distribution)
