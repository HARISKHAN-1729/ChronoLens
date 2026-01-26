"""
Success/Outcome feature extractor
Evaluates task completion, accuracy, efficiency, and self-corrections
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class SuccessOutcomeExtractor(FeatureExtractor):
    """Extract task success and outcome metrics"""
    
    def extract(self):
        """Extract all success/outcome features"""
        completion_indicators = []
        error_corrections = []
        efficiency_metrics = []
        goal_progress = []
        
        prev_frame = None
        task_start_time = 0
        corrections_count = 0
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=2):
            if prev_frame is not None:
                # Detect completion indicators
                completion = self._detect_completion_indicator(frame)
                if completion:
                    completion_indicators.append({
                        "timestamp": timestamp,
                        "type": completion,
                        "frame": frame_num
                    })
                
                # Detect self-corrections
                correction = self._detect_correction(frame, prev_frame)
                if correction:
                    error_corrections.append({
                        "timestamp": timestamp,
                        "correction_type": correction,
                        "frame": frame_num
                    })
                    corrections_count += 1
                
                # Track progress indicators
                progress = self._estimate_progress(frame, timestamp)
                if progress:
                    goal_progress.append(progress)
            
            prev_frame = frame.copy()
        
        # Calculate success metrics
        task_completed = len(completion_indicators) > 0
        completion_time = completion_indicators[-1]["timestamp"] if completion_indicators else self.duration
        
        # Calculate efficiency
        efficiency_score = self._calculate_efficiency(completion_time, corrections_count)
        
        # Calculate accuracy
        accuracy_score = self._calculate_accuracy(error_corrections, completion_indicators)
        
        return {
            "task_completed": task_completed,
            "completion_time": float(completion_time),
            "completion_rate": 1.0 if task_completed else self._estimate_completion_rate(goal_progress),
            "total_corrections": corrections_count,
            "correction_rate": float(corrections_count / (self.duration / 60)) if self.duration > 0 else 0.0,
            "efficiency_score": float(efficiency_score),
            "accuracy_score": float(accuracy_score),
            "success_score": float(self._calculate_success_score(task_completed, efficiency_score, accuracy_score)),
            "time_to_completion": float(completion_time),
            "optimal_path_deviation": self._calculate_path_deviation(error_corrections),
            "detailed_completion_indicators": completion_indicators[:20],
            "detailed_corrections": error_corrections[:50],
            "progress_timeline": goal_progress[:50]
        }
    
    def _detect_completion_indicator(self, frame):
        """Detect indicators of task completion"""
        # Use OCR to look for completion messages
        ocr_reader = self.model_loader.get_ocr_reader()
        results = ocr_reader.readtext(frame)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        completion_keywords = [
            "success", "complete", "done", "finished", "submitted",
            "confirmed", "thank you", "congratulations", "approved"
        ]
        
        for keyword in completion_keywords:
            if keyword in text_content:
                return f"completion_{keyword}"
        
        # Look for visual indicators (checkmarks, green colors)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range (success indicator)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_pixels = np.sum(green_mask > 0)
        if green_pixels > 1000:
            return "visual_success_indicator"
        
        return None
    
    def _detect_correction(self, frame, prev_frame):
        """Detect self-correction behavior"""
        # Look for undo/delete actions
        ocr_reader = self.model_loader.get_ocr_reader()
        results = ocr_reader.readtext(frame)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        correction_keywords = ["undo", "delete", "clear", "reset", "back"]
        
        for keyword in correction_keywords:
            if keyword in text_content:
                return f"correction_{keyword}"
        
        # Detect backspace/delete through content reduction
        frame_content = np.sum(frame > 128)
        prev_content = np.sum(prev_frame > 128)
        
        if prev_content > frame_content * 1.1:
            return "content_deletion"
        
        return None
    
    def _estimate_progress(self, frame, timestamp):
        """Estimate task progress"""
        # Look for progress indicators
        ocr_reader = self.model_loader.get_ocr_reader()
        results = ocr_reader.readtext(frame)
        
        for bbox, text, conf in results:
            # Look for percentage or step indicators
            if "%" in text:
                try:
                    percentage = float(text.replace("%", "").strip())
                    return {
                        "timestamp": timestamp,
                        "progress": percentage / 100.0
                    }
                except:
                    pass
            
            # Look for "step X of Y"
            if "step" in text.lower():
                return {
                    "timestamp": timestamp,
                    "progress": 0.5  # Estimate
                }
        
        return None
    
    def _calculate_efficiency(self, completion_time, corrections_count):
        """Calculate efficiency score (0-1)"""
        # Lower time and fewer corrections = higher efficiency
        
        # Normalize time (assume 5 minutes is baseline)
        time_score = max(0, 1 - (completion_time / 300))
        
        # Normalize corrections (assume 10 corrections is poor)
        correction_score = max(0, 1 - (corrections_count / 10))
        
        # Combine scores
        efficiency = (time_score * 0.6 + correction_score * 0.4)
        
        return efficiency
    
    def _calculate_accuracy(self, error_corrections, completion_indicators):
        """Calculate accuracy score (0-1)"""
        # Fewer corrections and successful completion = higher accuracy
        
        if not completion_indicators:
            return 0.5  # Incomplete task
        
        # Penalize for corrections
        correction_penalty = min(len(error_corrections) * 0.05, 0.5)
        
        accuracy = 1.0 - correction_penalty
        
        return max(0, accuracy)
    
    def _calculate_success_score(self, completed, efficiency, accuracy):
        """Calculate overall success score (0-1)"""
        if not completed:
            return 0.3  # Partial credit for attempt
        
        # Weighted combination
        success = (
            0.4 * 1.0 +  # Completion
            0.3 * efficiency +
            0.3 * accuracy
        )
        
        return success
    
    def _estimate_completion_rate(self, goal_progress):
        """Estimate completion rate from progress indicators"""
        if not goal_progress:
            return 0.5  # Unknown
        
        # Use last known progress
        last_progress = goal_progress[-1]["progress"]
        return float(last_progress)
    
    def _calculate_path_deviation(self, error_corrections):
        """Calculate deviation from optimal path"""
        # More corrections = more deviation
        if not error_corrections:
            return 0.0
        
        # Normalize to 0-1 scale
        deviation = min(len(error_corrections) / 20, 1.0)
        
        return float(deviation)
