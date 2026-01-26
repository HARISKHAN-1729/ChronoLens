#!/usr/bin/env python3
"""
Example: Create a custom feature extractor
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractors.base_extractor import FeatureExtractor

class CustomMetricsExtractor(FeatureExtractor):
    """
    Example custom feature extractor
    Extracts custom metrics like color distribution, brightness changes, etc.
    """
    
    def extract(self):
        """Extract custom features"""
        
        color_distribution = []
        brightness_changes = []
        prev_brightness = None
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=10):
            # Calculate average color
            avg_color = np.mean(frame, axis=(0, 1))
            color_distribution.append({
                "timestamp": timestamp,
                "blue": float(avg_color[0]),
                "green": float(avg_color[1]),
                "red": float(avg_color[2])
            })
            
            # Calculate brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if prev_brightness is not None:
                change = brightness - prev_brightness
                if abs(change) > 10:  # Significant change
                    brightness_changes.append({
                        "timestamp": timestamp,
                        "change": float(change)
                    })
            
            prev_brightness = brightness
        
        return {
            "average_red": float(np.mean([c["red"] for c in color_distribution])),
            "average_green": float(np.mean([c["green"] for c in color_distribution])),
            "average_blue": float(np.mean([c["blue"] for c in color_distribution])),
            "brightness_changes": len(brightness_changes),
            "significant_brightness_events": brightness_changes[:10]
        }

# Usage example
if __name__ == "__main__":
    from models.model_loader import ModelLoader
    
    video_path = "path/to/your/video.mp4"
    
    model_loader = ModelLoader()
    model_loader.load_all_models()
    
    with CustomMetricsExtractor(video_path, model_loader) as extractor:
        features = extractor.extract()
        print("Custom features extracted:")
        print(features)
