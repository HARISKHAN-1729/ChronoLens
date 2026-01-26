"""
Interaction Type feature extractor
Detects different types of user interactions: clicks, double-clicks, drags, scrolls, keyboard input
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import INTERACTION_CONFIG

class InteractionTypeExtractor(FeatureExtractor):
    """Extract and classify different interaction types"""
    
    def extract(self):
        """Extract all interaction type features"""
        clicks = []
        double_clicks = []
        drags = []
        scrolls = []
        keyboard_inputs = []
        
        prev_frame = None
        last_click_time = None
        drag_start = None
        in_drag = False
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=1):
            if prev_frame is not None:
                # Detect click events
                click_detected, click_pos = self._detect_click(frame, prev_frame)
                
                if click_detected:
                    clicks.append({
                        "timestamp": timestamp,
                        "position": click_pos,
                        "frame": frame_num
                    })
                    
                    # Check for double-click
                    if last_click_time and (timestamp - last_click_time) < INTERACTION_CONFIG["double_click_threshold"]:
                        double_clicks.append({
                            "timestamp": timestamp,
                            "position": click_pos,
                            "time_between_clicks": timestamp - last_click_time
                        })
                    
                    last_click_time = timestamp
                    
                    # Start potential drag
                    if not in_drag:
                        drag_start = {"time": timestamp, "pos": click_pos, "frame": frame_num}
                        in_drag = True
                
                # Detect drag continuation/end
                if in_drag:
                    motion = self._detect_cursor_motion(frame, prev_frame)
                    if motion:
                        distance = np.linalg.norm(np.array(motion) - np.array(drag_start["pos"]))
                        if distance > INTERACTION_CONFIG["drag_threshold"]:
                            # This is a drag
                            drags.append({
                                "start_time": drag_start["time"],
                                "end_time": timestamp,
                                "start_pos": drag_start["pos"],
                                "end_pos": motion,
                                "distance": float(distance),
                                "duration": timestamp - drag_start["time"]
                            })
                            in_drag = False
                            drag_start = None
                    else:
                        # No motion, end drag
                        in_drag = False
                        drag_start = None
                
                # Detect scroll events
                scroll_amount = self._detect_scroll(frame, prev_frame)
                if abs(scroll_amount) > INTERACTION_CONFIG["scroll_threshold"]:
                    scrolls.append({
                        "timestamp": timestamp,
                        "amount": float(scroll_amount),
                        "direction": "down" if scroll_amount > 0 else "up"
                    })
                
                # Detect keyboard input (text changes, UI updates)
                keyboard_detected = self._detect_keyboard_input(frame, prev_frame)
                if keyboard_detected:
                    keyboard_inputs.append({
                        "timestamp": timestamp,
                        "frame": frame_num
                    })
            
            prev_frame = frame.copy()
        
        # Detect copy-paste events (rapid text appearance)
        copy_paste_events = self._detect_copy_paste(keyboard_inputs)
        
        return {
            "total_clicks": len(clicks),
            "total_double_clicks": len(double_clicks),
            "total_drags": len(drags),
            "total_scrolls": len(scrolls),
            "total_keyboard_inputs": len(keyboard_inputs),
            "total_copy_paste_events": len(copy_paste_events),
            "click_rate_per_minute": len(clicks) / (self.duration / 60) if self.duration > 0 else 0.0,
            "average_drag_distance": float(np.mean([d["distance"] for d in drags])) if drags else 0.0,
            "average_drag_duration": float(np.mean([d["duration"] for d in drags])) if drags else 0.0,
            "scroll_direction_ratio": self._calculate_scroll_ratio(scrolls),
            "interaction_distribution": {
                "clicks": len(clicks),
                "double_clicks": len(double_clicks),
                "drags": len(drags),
                "scrolls": len(scrolls),
                "keyboard": len(keyboard_inputs)
            },
            "detailed_clicks": clicks[:50],
            "detailed_double_clicks": double_clicks[:50],
            "detailed_drags": drags[:50],
            "detailed_scrolls": scrolls[:50]
        }
    
    def _detect_click(self, frame, prev_frame):
        """Detect click event through visual feedback"""
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Look for sudden bright spots (click feedback)
        _, thresh = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Click feedback size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return True, (cx, cy)
        
        return False, None
    
    def _detect_cursor_motion(self, frame, prev_frame):
        """Detect cursor position for drag tracking"""
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 5000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        
        return None
    
    def _detect_scroll(self, frame, prev_frame):
        """Detect scrolling through content shift"""
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vertical_flow = np.mean(flow[:, :, 1])
            return vertical_flow
        except:
            return 0.0
    
    def _detect_keyboard_input(self, frame, prev_frame):
        """Detect keyboard input through text area changes"""
        # Focus on typical text input areas (center and lower portions)
        h, w = frame.shape[:2]
        roi_curr = frame[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]
        roi_prev = prev_frame[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]
        
        diff = cv2.absdiff(roi_curr, roi_prev)
        mean_diff = np.mean(diff)
        
        # Keyboard input causes subtle but consistent changes
        return 5 < mean_diff < 30
    
    def _detect_copy_paste(self, keyboard_inputs):
        """Detect copy-paste events (rapid text appearance)"""
        copy_paste = []
        
        for i in range(1, len(keyboard_inputs)):
            time_gap = keyboard_inputs[i]["timestamp"] - keyboard_inputs[i-1]["timestamp"]
            # Very short gap suggests paste rather than typing
            if time_gap < 0.1:
                copy_paste.append({
                    "timestamp": keyboard_inputs[i]["timestamp"],
                    "type": "paste"
                })
        
        return copy_paste
    
    def _calculate_scroll_ratio(self, scrolls):
        """Calculate ratio of scroll directions"""
        if not scrolls:
            return {"up": 0.0, "down": 0.0}
        
        up_count = sum(1 for s in scrolls if s["direction"] == "up")
        down_count = sum(1 for s in scrolls if s["direction"] == "down")
        total = len(scrolls)
        
        return {
            "up": up_count / total if total > 0 else 0.0,
            "down": down_count / total if total > 0 else 0.0
        }
