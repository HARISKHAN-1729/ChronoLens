"""
Technical Performance feature extractor
Detects lag, loading times, frame rate drops, and technical issues
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class TechnicalPerformanceExtractor(FeatureExtractor):
    """Extract technical performance metrics"""
    
    def extract(self):
        """Extract all technical performance features"""
        lag_events = []
        loading_times = []
        frame_drops = []
        performance_issues = []
        
        prev_frame = None
        prev_timestamp = None
        loading_start = None
        
        frame_times = []
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=0):
            # Track frame timing
            if prev_timestamp is not None:
                frame_time = timestamp - prev_timestamp
                frame_times.append(frame_time)
                
                # Detect frame drops (irregular timing)
                expected_frame_time = 1.0 / self.fps if self.fps > 0 else 0.033
                if frame_time > expected_frame_time * 2:
                    frame_drops.append({
                        "timestamp": timestamp,
                        "frame": frame_num,
                        "delay": float(frame_time - expected_frame_time)
                    })
            
            if prev_frame is not None:
                # Detect lag (frozen frames)
                similarity = self._calculate_frame_similarity(frame, prev_frame)
                
                if similarity > 0.99:  # Nearly identical frames
                    lag_events.append({
                        "timestamp": timestamp,
                        "frame": frame_num
                    })
                
                # Detect loading indicators
                loading_detected = self._detect_loading(frame)
                
                if loading_detected:
                    if loading_start is None:
                        loading_start = timestamp
                else:
                    if loading_start is not None:
                        loading_duration = timestamp - loading_start
                        loading_times.append({
                            "start_time": loading_start,
                            "end_time": timestamp,
                            "duration": loading_duration
                        })
                        loading_start = None
                
                # Detect other performance issues
                issue = self._detect_performance_issue(frame, prev_frame)
                if issue:
                    performance_issues.append({
                        "timestamp": timestamp,
                        "issue_type": issue,
                        "frame": frame_num
                    })
            
            prev_frame = frame.copy()
            prev_timestamp = timestamp
        
        # Calculate aggregate metrics
        avg_frame_time = np.mean(frame_times) if frame_times else 0.0
        frame_time_variance = np.var(frame_times) if frame_times else 0.0
        actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        total_loading_time = sum([l["duration"] for l in loading_times])
        avg_loading_time = np.mean([l["duration"] for l in loading_times]) if loading_times else 0.0
        
        return {
            "total_lag_events": len(lag_events),
            "lag_rate_per_minute": float(len(lag_events) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "total_loading_events": len(loading_times),
            "total_loading_time": float(total_loading_time),
            "average_loading_time": float(avg_loading_time),
            "total_frame_drops": len(frame_drops),
            "frame_drop_rate": float(len(frame_drops) / self.total_frames) if self.total_frames > 0 else 0.0,
            "average_frame_time": float(avg_frame_time),
            "frame_time_variance": float(frame_time_variance),
            "actual_fps": float(actual_fps),
            "expected_fps": float(self.fps),
            "fps_stability": self._calculate_fps_stability(frame_times),
            "performance_score": self._calculate_performance_score(lag_events, loading_times, frame_drops),
            "total_performance_issues": len(performance_issues),
            "detailed_lag_events": lag_events[:50],
            "detailed_loading_times": loading_times[:30],
            "detailed_frame_drops": frame_drops[:50],
            "detailed_performance_issues": performance_issues[:50]
        }
    
    def _calculate_frame_similarity(self, frame1, frame2):
        """Calculate similarity between consecutive frames"""
        # Use structural similarity
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Simple correlation-based similarity
        correlation = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
        
        return abs(correlation)
    
    def _detect_loading(self, frame):
        """Detect loading indicators"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect loading spinners (circular patterns)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            return True
        
        # Detect progress bars
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
            if len(horizontal_lines) >= 2:
                return True
        
        # Check for "loading" text
        try:
            ocr_reader = self.model_loader.get_ocr_reader()
            results = ocr_reader.readtext(frame)
            
            text_content = " ".join([text for _, text, _ in results]).lower()
            
            if "loading" in text_content or "please wait" in text_content:
                return True
        except:
            pass
        
        return False
    
    def _detect_performance_issue(self, frame, prev_frame):
        """Detect various performance issues"""
        # Detect screen tearing (horizontal artifacts)
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Check for horizontal bands of high difference
        row_diffs = np.mean(gray_diff, axis=1)
        
        if np.max(row_diffs) > 100 and np.std(row_diffs) > 30:
            return "screen_tearing"
        
        # Detect low frame rate (blurry motion)
        laplacian = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        if blur_score < 100:
            return "motion_blur"
        
        # Detect compression artifacts
        if self._has_compression_artifacts(frame):
            return "compression_artifacts"
        
        return None
    
    def _has_compression_artifacts(self, frame):
        """Detect compression artifacts"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for blocking artifacts (8x8 patterns)
        # Simplified check
        block_size = 8
        h, w = gray.shape
        
        block_variances = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))
        
        # High variance in block variances suggests artifacts
        if block_variances:
            variance_of_variances = np.var(block_variances)
            return variance_of_variances > 1000
        
        return False
    
    def _calculate_fps_stability(self, frame_times):
        """Calculate FPS stability score (0-1)"""
        if len(frame_times) < 2:
            return 1.0
        
        # Low variance = high stability
        variance = np.var(frame_times)
        mean = np.mean(frame_times)
        
        # Coefficient of variation
        cv = variance / (mean ** 2) if mean > 0 else 0
        
        # Convert to stability score
        stability = 1 - min(cv * 10, 1.0)
        
        return float(stability)
    
    def _calculate_performance_score(self, lag_events, loading_times, frame_drops):
        """Calculate overall performance score (0-1)"""
        # Penalize for issues
        lag_penalty = min(len(lag_events) / 100, 0.3)
        loading_penalty = min(len(loading_times) / 20, 0.3)
        drop_penalty = min(len(frame_drops) / 50, 0.4)
        
        score = 1.0 - (lag_penalty + loading_penalty + drop_penalty)
        
        return float(max(0, score))
