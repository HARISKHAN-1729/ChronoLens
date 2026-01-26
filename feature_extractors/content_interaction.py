"""
Content Interaction feature extractor
Analyzes text input patterns, typing speed, form completion, media interactions
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class ContentInteractionExtractor(FeatureExtractor):
    """Extract content interaction features"""
    
    def extract(self):
        """Extract all content interaction features"""
        text_inputs = []
        form_interactions = []
        media_interactions = []
        
        prev_frame = None
        typing_sessions = []
        current_session = None
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=1):
            if prev_frame is not None:
                # Detect text input activity
                text_change = self._detect_text_change(frame, prev_frame)
                
                if text_change:
                    text_inputs.append({
                        "timestamp": timestamp,
                        "frame": frame_num
                    })
                    
                    # Track typing session
                    if current_session is None:
                        current_session = {
                            "start_time": timestamp,
                            "keystrokes": 1
                        }
                    else:
                        current_session["keystrokes"] += 1
                else:
                    # End typing session if gap detected
                    if current_session and (timestamp - text_inputs[-1]["timestamp"] if text_inputs else 0) > 2.0:
                        current_session["end_time"] = text_inputs[-1]["timestamp"] if text_inputs else timestamp
                        current_session["duration"] = current_session["end_time"] - current_session["start_time"]
                        typing_sessions.append(current_session)
                        current_session = None
                
                # Detect form field interactions
                form_activity = self._detect_form_interaction(frame, prev_frame)
                if form_activity:
                    form_interactions.append({
                        "timestamp": timestamp,
                        "activity_type": form_activity
                    })
                
                # Detect media interactions (play, pause, seek)
                media_activity = self._detect_media_interaction(frame, prev_frame)
                if media_activity:
                    media_interactions.append({
                        "timestamp": timestamp,
                        "interaction_type": media_activity
                    })
            
            prev_frame = frame.copy()
        
        # Close any open typing session
        if current_session:
            current_session["end_time"] = self.duration
            current_session["duration"] = current_session["end_time"] - current_session["start_time"]
            typing_sessions.append(current_session)
        
        # Calculate typing metrics
        typing_speed = self._calculate_typing_speed(typing_sessions)
        
        return {
            "total_text_inputs": len(text_inputs),
            "total_typing_sessions": len(typing_sessions),
            "average_typing_speed_cpm": float(typing_speed),
            "total_form_interactions": len(form_interactions),
            "total_media_interactions": len(media_interactions),
            "average_session_duration": float(np.mean([s["duration"] for s in typing_sessions])) if typing_sessions else 0.0,
            "total_keystrokes": sum(s["keystrokes"] for s in typing_sessions),
            "form_completion_rate": self._estimate_form_completion(form_interactions),
            "text_input_rate_per_minute": float(len(text_inputs) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "detailed_typing_sessions": typing_sessions[:20],
            "detailed_form_interactions": form_interactions[:50],
            "detailed_media_interactions": media_interactions[:50],
            "interaction_distribution": {
                "text_input": len(text_inputs),
                "form": len(form_interactions),
                "media": len(media_interactions)
            }
        }
    
    def _detect_text_change(self, frame, prev_frame):
        """Detect text input through content changes"""
        # Focus on typical text input areas
        h, w = frame.shape[:2]
        
        # Sample multiple regions for text input
        regions = [
            (int(h*0.3), int(h*0.7), int(w*0.1), int(w*0.9)),  # Center
            (int(h*0.1), int(h*0.3), int(w*0.1), int(w*0.9)),  # Top
            (int(h*0.7), int(h*0.9), int(w*0.1), int(w*0.9))   # Bottom
        ]
        
        for y1, y2, x1, x2 in regions:
            roi_curr = frame[y1:y2, x1:x2]
            roi_prev = prev_frame[y1:y2, x1:x2]
            
            diff = cv2.absdiff(roi_curr, roi_prev)
            mean_diff = np.mean(diff)
            
            # Text changes are subtle but consistent
            if 5 < mean_diff < 25:
                return True
        
        return False
    
    def _detect_form_interaction(self, frame, prev_frame):
        """Detect form field interactions"""
        # Look for cursor focus changes (field highlighting)
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Check for border/highlight changes
        edges_curr = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_prev = cv2.Canny(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), 50, 150)
        
        edge_diff = cv2.absdiff(edges_curr, edges_prev)
        edge_change = np.sum(edge_diff > 0)
        
        if edge_change > 1000:
            return "field_focus_change"
        
        # Check for checkbox/radio button clicks
        if np.max(gray_diff) > 100:
            return "checkbox_toggle"
        
        return None
    
    def _detect_media_interaction(self, frame, prev_frame):
        """Detect media player interactions"""
        # Look for play/pause button area changes
        h, w = frame.shape[:2]
        
        # Typical media control area (bottom center)
        control_area_curr = frame[int(h*0.85):h, int(w*0.4):int(w*0.6)]
        control_area_prev = prev_frame[int(h*0.85):h, int(w*0.4):int(w*0.6)]
        
        if control_area_curr.size == 0 or control_area_prev.size == 0:
            return None
        
        diff = cv2.absdiff(control_area_curr, control_area_prev)
        mean_diff = np.mean(diff)
        
        if mean_diff > 30:
            return "play_pause"
        
        # Check for seek bar interaction (horizontal changes)
        seek_area_curr = frame[int(h*0.9):int(h*0.95), int(w*0.1):int(w*0.9)]
        seek_area_prev = prev_frame[int(h*0.9):int(h*0.95), int(w*0.1):int(w*0.9)]
        
        if seek_area_curr.size > 0 and seek_area_prev.size > 0:
            seek_diff = cv2.absdiff(seek_area_curr, seek_area_prev)
            if np.mean(seek_diff) > 20:
                return "seek"
        
        return None
    
    def _calculate_typing_speed(self, typing_sessions):
        """Calculate average typing speed in characters per minute"""
        if not typing_sessions:
            return 0.0
        
        total_keystrokes = sum(s["keystrokes"] for s in typing_sessions)
        total_time = sum(s["duration"] for s in typing_sessions)
        
        if total_time > 0:
            return (total_keystrokes / total_time) * 60
        
        return 0.0
    
    def _estimate_form_completion(self, form_interactions):
        """Estimate form completion rate"""
        if not form_interactions:
            return 0.0
        
        # Heuristic: more interactions suggest more completion
        # This is a simplified estimate
        field_changes = sum(1 for f in form_interactions if f["activity_type"] == "field_focus_change")
        
        # Assume average form has 5 fields
        estimated_fields = 5
        completion = min(field_changes / estimated_fields, 1.0)
        
        return float(completion)
