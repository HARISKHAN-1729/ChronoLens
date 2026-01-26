"""
Emotional/Affective feature extractor
Detects frustration, engagement, confusion through interaction patterns
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class EmotionalAffectiveExtractor(FeatureExtractor):
    """Extract emotional and affective state indicators"""
    
    def extract(self):
        """Extract all emotional/affective features"""
        frustration_events = []
        engagement_indicators = []
        confusion_events = []
        satisfaction_indicators = []
        
        prev_frame = None
        click_history = []
        pause_history = []
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=1):
            if prev_frame is not None:
                # Detect frustration indicators
                frustration = self._detect_frustration(frame, prev_frame, click_history)
                if frustration:
                    frustration_events.append({
                        "timestamp": timestamp,
                        "type": frustration,
                        "intensity": self._calculate_frustration_intensity(click_history)
                    })
                
                # Detect engagement level
                engagement = self._detect_engagement(frame, prev_frame)
                if engagement:
                    engagement_indicators.append({
                        "timestamp": timestamp,
                        "level": engagement
                    })
                
                # Detect confusion
                confusion = self._detect_confusion(frame, prev_frame, pause_history)
                if confusion:
                    confusion_events.append({
                        "timestamp": timestamp,
                        "indicators": confusion
                    })
                
                # Detect satisfaction
                satisfaction = self._detect_satisfaction(frame)
                if satisfaction:
                    satisfaction_indicators.append({
                        "timestamp": timestamp,
                        "type": satisfaction
                    })
                
                # Track clicks for frustration detection
                click_detected = self._detect_click(frame, prev_frame)
                if click_detected:
                    click_history.append(timestamp)
                    # Keep only recent clicks
                    click_history = [t for t in click_history if timestamp - t < 5.0]
                
                # Track pauses
                if self._is_paused(frame, prev_frame):
                    if not pause_history or timestamp - pause_history[-1] > 1.0:
                        pause_history.append(timestamp)
            
            prev_frame = frame.copy()
        
        # Calculate aggregate emotional metrics
        avg_frustration = self._calculate_avg_frustration(frustration_events)
        avg_engagement = self._calculate_avg_engagement(engagement_indicators)
        emotional_variance = self._calculate_emotional_variance(frustration_events, satisfaction_indicators)
        
        return {
            "total_frustration_events": len(frustration_events),
            "average_frustration_intensity": float(avg_frustration),
            "frustration_rate_per_minute": float(len(frustration_events) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "total_confusion_events": len(confusion_events),
            "confusion_rate_per_minute": float(len(confusion_events) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "average_engagement_level": float(avg_engagement),
            "total_satisfaction_indicators": len(satisfaction_indicators),
            "emotional_variance": float(emotional_variance),
            "overall_sentiment": self._classify_overall_sentiment(frustration_events, satisfaction_indicators),
            "detailed_frustration_events": frustration_events[:50],
            "detailed_confusion_events": confusion_events[:50],
            "detailed_satisfaction_indicators": satisfaction_indicators[:30],
            "engagement_timeline": engagement_indicators[:100]
        }
    
    def _detect_frustration(self, frame, prev_frame, click_history):
        """Detect frustration through rapid clicking and erratic behavior"""
        # Rapid clicking in short time window
        if len(click_history) >= 5:
            time_span = click_history[-1] - click_history[0]
            if time_span < 2.0:  # 5 clicks in 2 seconds
                return "rapid_clicking"
        
        # Detect aggressive mouse movements (large frame differences)
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > 60:
            return "aggressive_interaction"
        
        # Detect repeated actions in same area (stuck behavior)
        if len(click_history) >= 3:
            # Simplified check
            return None
        
        return None
    
    def _detect_engagement(self, frame, prev_frame):
        """Detect engagement level through activity patterns"""
        # Calculate activity level
        diff = cv2.absdiff(frame, prev_frame)
        activity = np.mean(diff)
        
        # Classify engagement
        if activity > 30:
            return "high"
        elif activity > 10:
            return "medium"
        elif activity > 2:
            return "low"
        else:
            return "disengaged"
    
    def _detect_confusion(self, frame, prev_frame, pause_history):
        """Detect confusion through hesitation and backtracking"""
        # Long pauses indicate confusion
        if len(pause_history) >= 2:
            recent_pauses = [p for p in pause_history if self.duration - p < 30]
            if len(recent_pauses) >= 3:
                return ["frequent_pauses"]
        
        # Minimal activity (staring at screen)
        diff = cv2.absdiff(frame, prev_frame)
        if np.mean(diff) < 3:
            return ["minimal_activity"]
        
        return None
    
    def _detect_satisfaction(self, frame):
        """Detect satisfaction indicators"""
        # Look for success messages or positive UI feedback
        ocr_reader = self.model_loader.get_ocr_reader()
        results = ocr_reader.readtext(frame)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        positive_keywords = [
            "success", "complete", "done", "great", "excellent",
            "congratulations", "thank you", "approved"
        ]
        
        for keyword in positive_keywords:
            if keyword in text_content:
                return f"positive_{keyword}"
        
        # Look for green/positive colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        if np.sum(green_mask > 0) > 1000:
            return "positive_visual_feedback"
        
        return None
    
    def _detect_click(self, frame, prev_frame):
        """Detect click event"""
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                return True
        
        return False
    
    def _is_paused(self, frame, prev_frame):
        """Check if user is paused/idle"""
        diff = cv2.absdiff(frame, prev_frame)
        return np.mean(diff) < 5
    
    def _calculate_frustration_intensity(self, click_history):
        """Calculate frustration intensity from click patterns"""
        if len(click_history) < 2:
            return 0.0
        
        # More clicks in shorter time = higher intensity
        time_span = click_history[-1] - click_history[0]
        if time_span > 0:
            click_rate = len(click_history) / time_span
            intensity = min(click_rate / 5, 1.0)  # Normalize
            return intensity
        
        return 0.0
    
    def _calculate_avg_frustration(self, frustration_events):
        """Calculate average frustration intensity"""
        if not frustration_events:
            return 0.0
        
        intensities = [e["intensity"] for e in frustration_events]
        return np.mean(intensities)
    
    def _calculate_avg_engagement(self, engagement_indicators):
        """Calculate average engagement level"""
        if not engagement_indicators:
            return 0.5
        
        level_map = {"disengaged": 0.0, "low": 0.33, "medium": 0.66, "high": 1.0}
        
        levels = [level_map.get(e["level"], 0.5) for e in engagement_indicators]
        return np.mean(levels)
    
    def _calculate_emotional_variance(self, frustration_events, satisfaction_indicators):
        """Calculate variance in emotional state"""
        # High variance indicates emotional instability
        emotional_scores = []
        
        # Frustration = negative scores
        for event in frustration_events:
            emotional_scores.append(-event["intensity"])
        
        # Satisfaction = positive scores
        for _ in satisfaction_indicators:
            emotional_scores.append(1.0)
        
        if emotional_scores:
            return np.var(emotional_scores)
        
        return 0.0
    
    def _classify_overall_sentiment(self, frustration_events, satisfaction_indicators):
        """Classify overall emotional sentiment"""
        frustration_count = len(frustration_events)
        satisfaction_count = len(satisfaction_indicators)
        
        if satisfaction_count > frustration_count * 2:
            return "positive"
        elif frustration_count > satisfaction_count * 2:
            return "negative"
        else:
            return "neutral"
