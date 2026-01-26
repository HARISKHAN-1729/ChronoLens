"""
Cognitive Load feature extractor
Measures decision time, help-seeking behavior, trial-and-error patterns
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import COGNITIVE_LOAD_CONFIG

class CognitiveLoadExtractor(FeatureExtractor):
    """Extract cognitive load indicators"""
    
    def extract(self):
        """Extract all cognitive load features"""
        decision_points = []
        help_seeking_events = []
        trial_error_patterns = []
        hesitation_events = []
        action_frequency = []
        
        prev_frame = None
        last_action_time = None
        idle_start = None
        action_history = []
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=1):
            if prev_frame is not None:
                # Detect user actions
                action_detected = self._detect_action(frame, prev_frame)
                
                if action_detected:
                    # Calculate time since last action
                    if last_action_time:
                        time_gap = timestamp - last_action_time
                        
                        # Long gap indicates decision/thinking time
                        if time_gap > COGNITIVE_LOAD_CONFIG["decision_threshold"]:
                            decision_points.append({
                                "timestamp": timestamp,
                                "decision_time": time_gap,
                                "frame": frame_num
                            })
                        
                        # Track action frequency
                        action_frequency.append({
                            "timestamp": timestamp,
                            "time_since_last": time_gap
                        })
                    
                    last_action_time = timestamp
                    action_history.append({
                        "timestamp": timestamp,
                        "type": action_detected
                    })
                    
                    # Keep recent history
                    if len(action_history) > 50:
                        action_history.pop(0)
                    
                    idle_start = None
                else:
                    # Track idle time
                    if idle_start is None:
                        idle_start = timestamp
                    else:
                        idle_duration = timestamp - idle_start
                        
                        # Extended idle suggests confusion or help-seeking
                        if idle_duration > COGNITIVE_LOAD_CONFIG["high_load_threshold"]:
                            hesitation_events.append({
                                "timestamp": idle_start,
                                "duration": idle_duration
                            })
                
                # Detect help-seeking behavior
                help_seeking = self._detect_help_seeking(frame, prev_frame)
                if help_seeking:
                    help_seeking_events.append({
                        "timestamp": timestamp,
                        "type": help_seeking,
                        "frame": frame_num
                    })
                
                # Detect trial-and-error patterns
                trial_error = self._detect_trial_error(action_history)
                if trial_error:
                    trial_error_patterns.append({
                        "timestamp": timestamp,
                        "pattern": trial_error
                    })
            
            prev_frame = frame.copy()
        
        # Calculate cognitive load metrics
        avg_decision_time = np.mean([d["decision_time"] for d in decision_points]) if decision_points else 0.0
        action_rate = self._calculate_action_rate(action_frequency)
        cognitive_load_score = self._calculate_load_score(decision_points, hesitation_events, trial_error_patterns)
        
        return {
            "total_decision_points": len(decision_points),
            "average_decision_time": float(avg_decision_time),
            "max_decision_time": float(max([d["decision_time"] for d in decision_points])) if decision_points else 0.0,
            "total_help_seeking_events": len(help_seeking_events),
            "total_trial_error_patterns": len(trial_error_patterns),
            "total_hesitation_events": len(hesitation_events),
            "average_hesitation_duration": float(np.mean([h["duration"] for h in hesitation_events])) if hesitation_events else 0.0,
            "action_rate_per_minute": float(action_rate),
            "cognitive_load_score": float(cognitive_load_score),
            "cognitive_load_level": self._classify_load_level(cognitive_load_score),
            "detailed_decision_points": decision_points[:50],
            "detailed_help_seeking": help_seeking_events[:30],
            "detailed_trial_error": trial_error_patterns[:30],
            "action_frequency_timeline": action_frequency[:100]
        }
    
    def _detect_action(self, frame, prev_frame):
        """Detect user action in frame"""
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        if mean_diff > 15:
            # Classify action type
            if mean_diff > 50:
                return "major_action"
            else:
                return "minor_action"
        
        return None
    
    def _detect_help_seeking(self, frame, prev_frame):
        """Detect help-seeking behavior"""
        # Look for help menu, search, or documentation access
        ocr_reader = self.model_loader.get_ocr_reader()
        
        # Focus on top menu area
        top_area = frame[:150, :]
        results = ocr_reader.readtext(top_area)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        help_keywords = ["help", "support", "tutorial", "guide", "faq", "search", "?"]
        
        for keyword in help_keywords:
            if keyword in text_content:
                return f"help_{keyword}"
        
        return None
    
    def _detect_trial_error(self, action_history):
        """Detect trial-and-error patterns"""
        if len(action_history) < 5:
            return None
        
        # Look for repeated similar actions in short time
        recent_actions = action_history[-5:]
        timestamps = [a["timestamp"] for a in recent_actions]
        
        # Check if actions are clustered (rapid attempts)
        time_span = timestamps[-1] - timestamps[0]
        
        if time_span < 5.0:  # 5 actions in 5 seconds
            return "rapid_attempts"
        
        # Check for alternating patterns (back and forth)
        action_types = [a["type"] for a in recent_actions]
        if len(set(action_types)) == 2:
            # Alternating between two actions
            return "alternating_attempts"
        
        return None
    
    def _calculate_action_rate(self, action_frequency):
        """Calculate actions per minute"""
        if not action_frequency or self.duration == 0:
            return 0.0
        
        return (len(action_frequency) / self.duration) * 60
    
    def _calculate_load_score(self, decision_points, hesitation_events, trial_error_patterns):
        """Calculate overall cognitive load score (0-1)"""
        # Normalize factors
        decision_factor = min(len(decision_points) / 20, 1.0)  # Normalize to 20 decisions
        hesitation_factor = min(len(hesitation_events) / 10, 1.0)  # Normalize to 10 hesitations
        trial_error_factor = min(len(trial_error_patterns) / 15, 1.0)  # Normalize to 15 patterns
        
        # Weighted combination
        load_score = (
            decision_factor * 0.3 +
            hesitation_factor * 0.4 +
            trial_error_factor * 0.3
        )
        
        return load_score
    
    def _classify_load_level(self, score):
        """Classify cognitive load level"""
        if score < 0.3:
            return "low"
        elif score < 0.6:
            return "medium"
        else:
            return "high"
