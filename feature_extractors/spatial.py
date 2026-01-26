"""
Spatial/Visual feature extractor
Extracts cursor movement, click locations, and screen region data
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import SPATIAL_CONFIG

class SpatialFeatureExtractor(FeatureExtractor):
    """Extract spatial and visual interaction features"""
    
    def extract(self):
        """Extract all spatial features"""
        cursor_positions = []
        click_locations = []
        screen_regions = defaultdict(int)
        scroll_events = []
        
        prev_frame = None
        prev_centroid = None
        
        grid_rows, grid_cols = SPATIAL_CONFIG["grid_size"]
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=2):
            if prev_frame is not None:
                # Detect cursor/pointer location using motion detection
                cursor_pos = self._detect_cursor_position(frame, prev_frame)
                
                if cursor_pos is not None:
                    x, y = cursor_pos
                    cursor_positions.append({
                        "timestamp": timestamp,
                        "x": int(x),
                        "y": int(y)
                    })
                    
                    # Determine screen region (grid-based)
                    region_x = int((x / self.width) * grid_cols)
                    region_y = int((y / self.height) * grid_rows)
                    region_key = f"{region_x}_{region_y}"
                    screen_regions[region_key] += 1
                    
                    # Detect clicks (sudden changes in motion patterns)
                    if self._is_click_event(frame, prev_frame, cursor_pos):
                        click_locations.append({
                            "timestamp": timestamp,
                            "x": int(x),
                            "y": int(y)
                        })
                
                # Detect scrolling
                scroll_delta = self._detect_scroll(frame, prev_frame)
                if abs(scroll_delta) > SPATIAL_CONFIG["scroll_threshold"]:
                    scroll_events.append({
                        "timestamp": timestamp,
                        "delta_pixels": float(scroll_delta),
                        "direction": "down" if scroll_delta > 0 else "up"
                    })
            
            prev_frame = frame.copy()
        
        # Calculate trajectory metrics
        trajectory_length = self._calculate_trajectory_length(cursor_positions)
        avg_cursor_speed = self._calculate_avg_speed(cursor_positions)
        
        # Generate heatmap data
        heatmap = self._generate_heatmap(cursor_positions)
        
        # Identify most visited regions
        most_visited = sorted(screen_regions.items(), key=lambda x: x[1], reverse=True)[:5]
        most_visited_regions = [{"region": k, "visits": v} for k, v in most_visited]
        
        # Calculate scrolling metrics
        total_scroll_distance = sum([abs(s["delta_pixels"]) for s in scroll_events])
        
        return {
            "cursor_trajectory_length_pixels": float(trajectory_length),
            "average_cursor_speed_pixels_per_sec": float(avg_cursor_speed),
            "total_cursor_positions": len(cursor_positions),
            "click_locations": click_locations[:100],  # Limit for JSON size
            "total_clicks": len(click_locations),
            "heatmap_data": heatmap,
            "most_visited_regions": most_visited_regions,
            "total_screen_regions_visited": len(screen_regions),
            "scroll_events": len(scroll_events),
            "total_scroll_distance_pixels": float(total_scroll_distance),
            "scroll_direction_distribution": self._get_scroll_distribution(scroll_events),
            "cursor_trajectory_sample": cursor_positions[:100]  # Sample for visualization
        }
    
    def _detect_cursor_position(self, frame, prev_frame):
        """Detect cursor/pointer position using motion analysis"""
        # Calculate frame difference
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find areas of change
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the most prominent small change (likely cursor)
        cursor_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Cursor is typically small but visible
            if 10 < area < 5000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cursor_candidates.append((cx, cy, area))
        
        if cursor_candidates:
            # Return the centroid of the largest small change
            cursor_candidates.sort(key=lambda x: x[2], reverse=True)
            return (cursor_candidates[0][0], cursor_candidates[0][1])
        
        return None
    
    def _is_click_event(self, frame, prev_frame, cursor_pos):
        """Detect if a click event occurred"""
        # Simple heuristic: look for sudden bright spot near cursor
        x, y = cursor_pos
        
        # Extract region around cursor
        region_size = 50
        x1, y1 = max(0, x - region_size), max(0, y - region_size)
        x2, y2 = min(self.width, x + region_size), min(self.height, y + region_size)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        region_curr = frame[y1:y2, x1:x2]
        region_prev = prev_frame[y1:y2, x1:x2]
        
        # Calculate intensity change
        diff = cv2.absdiff(region_curr, region_prev)
        mean_change = np.mean(diff)
        
        # Click often causes visual feedback
        return mean_change > 30
    
    def _detect_scroll(self, frame, prev_frame):
        """Detect scrolling by analyzing vertical content shift"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use phase correlation to detect vertical shift
        try:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Average vertical flow
            vertical_flow = np.mean(flow[:, :, 1])
            return vertical_flow
        except:
            return 0.0
    
    def _calculate_trajectory_length(self, positions):
        """Calculate total cursor trajectory length"""
        if len(positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]["x"], positions[i-1]["y"]
            x2, y2 = positions[i]["x"], positions[i]["y"]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += distance
        
        return total_length
    
    def _calculate_avg_speed(self, positions):
        """Calculate average cursor speed"""
        if len(positions) < 2:
            return 0.0
        
        speeds = []
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]["x"], positions[i-1]["y"]
            t1 = positions[i-1]["timestamp"]
            x2, y2 = positions[i]["x"], positions[i]["y"]
            t2 = positions[i]["timestamp"]
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_delta = t2 - t1
            
            if time_delta > 0:
                speed = distance / time_delta
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    def _generate_heatmap(self, positions):
        """Generate heatmap data from cursor positions"""
        grid_rows, grid_cols = SPATIAL_CONFIG["grid_size"]
        heatmap = np.zeros((grid_rows, grid_cols))
        
        for pos in positions:
            x, y = pos["x"], pos["y"]
            grid_x = int((x / self.width) * grid_cols)
            grid_y = int((y / self.height) * grid_rows)
            
            grid_x = min(grid_x, grid_cols - 1)
            grid_y = min(grid_y, grid_rows - 1)
            
            heatmap[grid_y, grid_x] += 1
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap.tolist()
    
    def _get_scroll_distribution(self, scroll_events):
        """Get distribution of scroll directions"""
        up_count = sum(1 for s in scroll_events if s["direction"] == "up")
        down_count = sum(1 for s in scroll_events if s["direction"] == "down")
        
        return {
            "up": up_count,
            "down": down_count
        }
