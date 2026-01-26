"""
Base class for all feature extractors
"""
from abc import ABC, abstractmethod
import cv2
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor(ABC):
    """Base class for all feature extraction modules"""
    
    def __init__(self, video_path, model_loader):
        """
        Initialize feature extractor
        
        Args:
            video_path: Path to video file
            model_loader: ModelLoader instance with loaded models
        """
        self.video_path = Path(video_path)
        self.model_loader = model_loader
        
        # Video properties
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Duration: {self.duration:.2f}s")
        logger.info(f"  Total frames: {self.total_frames}")
    
    @abstractmethod
    def extract(self):
        """
        Extract features from video
        
        Returns:
            dict: Extracted features
        """
        pass
    
    def frame_to_timestamp(self, frame_number):
        """Convert frame number to timestamp in seconds"""
        return frame_number / self.fps if self.fps > 0 else 0
    
    def timestamp_to_frame(self, timestamp):
        """Convert timestamp to frame number"""
        return int(timestamp * self.fps)
    
    def read_frame_at_time(self, timestamp):
        """Read a specific frame at given timestamp"""
        frame_number = self.timestamp_to_frame(timestamp)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def iterate_frames(self, skip_frames=1):
        """
        Iterator for video frames
        
        Args:
            skip_frames: Process every Nth frame (default: 1 = all frames)
            
        Yields:
            tuple: (frame_number, timestamp, frame)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_number % skip_frames == 0:
                timestamp = self.frame_to_timestamp(frame_number)
                yield frame_number, timestamp, frame
            
            frame_number += 1
    
    def calculate_frame_diff(self, frame1, frame2):
        """Calculate difference between two frames"""
        if frame1 is None or frame2 is None:
            return 0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Return mean difference
        return np.mean(diff)
    
    def detect_motion(self, frame1, frame2, threshold=10.0):
        """Detect if there's significant motion between frames"""
        diff = self.calculate_frame_diff(frame1, frame2)
        return diff > threshold
    
    def release(self):
        """Release video capture resources"""
        if self.cap:
            self.cap.release()
            logger.debug(f"Released video: {self.video_path.name}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False
