"""
UI State feature extractor
Uses YOLO and OCR to detect UI elements, error messages, loading states
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class UIStateExtractor(FeatureExtractor):
    """Extract UI state and element detection features"""
    
    def extract(self):
        """Extract all UI state features"""
        ui_elements = []
        error_messages = []
        loading_states = []
        button_clicks = []
        text_content = []
        
        yolo_model = self.model_loader.get_yolo_model()
        ocr_reader = self.model_loader.get_ocr_reader()
        
        prev_text = ""
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=3):
            # Detect UI elements with YOLO
            results = yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:
                        ui_elements.append({
                            "timestamp": timestamp,
                            "element_type": yolo_model.names[cls],
                            "confidence": conf,
                            "frame": frame_num
                        })
            
            # Perform OCR to detect text and error messages
            ocr_results = ocr_reader.readtext(frame)
            
            current_text = ""
            for detection in ocr_results:
                bbox, text, conf = detection
                current_text += text + " "
                
                text_content.append({
                    "timestamp": timestamp,
                    "text": text,
                    "confidence": float(conf)
                })
                
                # Detect error messages
                if self._is_error_message(text):
                    error_messages.append({
                        "timestamp": timestamp,
                        "message": text,
                        "frame": frame_num
                    })
            
            # Detect loading states
            loading_detected = self._detect_loading_state(frame)
            if loading_detected:
                loading_states.append({
                    "timestamp": timestamp,
                    "frame": frame_num,
                    "type": loading_detected
                })
            
            # Detect button states and clicks
            button_state = self._detect_button_interaction(frame, ui_elements)
            if button_state:
                button_clicks.append({
                    "timestamp": timestamp,
                    "button_type": button_state
                })
            
            prev_text = current_text
        
        # Aggregate metrics
        ui_element_distribution = self._get_element_distribution(ui_elements)
        
        return {
            "total_ui_elements_detected": len(ui_elements),
            "ui_element_distribution": ui_element_distribution,
            "total_error_messages": len(error_messages),
            "total_loading_states": len(loading_states),
            "total_button_clicks": len(button_clicks),
            "total_text_detected": len(text_content),
            "error_rate_per_minute": float(len(error_messages) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "loading_time_total": float(len(loading_states) * 0.1),  # Estimate
            "unique_ui_elements": len(set(e["element_type"] for e in ui_elements)),
            "detailed_ui_elements": ui_elements[:50],
            "detailed_error_messages": error_messages[:20],
            "detailed_loading_states": loading_states[:20],
            "detailed_button_clicks": button_clicks[:50],
            "text_content_sample": text_content[:30]
        }
    
    def _is_error_message(self, text):
        """Determine if text is an error message"""
        error_keywords = [
            "error", "failed", "invalid", "wrong", "incorrect",
            "cannot", "unable", "denied", "forbidden", "not found",
            "exception", "warning", "alert"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in error_keywords)
    
    def _detect_loading_state(self, frame):
        """Detect loading indicators"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect circular patterns (loading spinners)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            return "spinner"
        
        # Detect progress bars (horizontal rectangles with gradient)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
            if len(horizontal_lines) > 2:
                return "progress_bar"
        
        return None
    
    def _detect_button_interaction(self, frame, ui_elements):
        """Detect button interactions"""
        # Look for button elements in recent detections
        recent_buttons = [e for e in ui_elements[-5:] if "button" in e.get("element_type", "").lower()]
        
        if recent_buttons:
            return recent_buttons[-1]["element_type"]
        
        return None
    
    def _get_element_distribution(self, ui_elements):
        """Get distribution of UI element types"""
        distribution = defaultdict(int)
        
        for element in ui_elements:
            distribution[element["element_type"]] += 1
        
        return dict(distribution)
