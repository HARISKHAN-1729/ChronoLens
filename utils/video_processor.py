"""
Video processing utilities
"""
import cv2
import logging
from pathlib import Path
from config import VALIDATION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_video_files(folder_path):
    """
    Get all video files from a folder
    
    Args:
        folder_path: Path to folder containing videos
        
    Returns:
        List of video file paths
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    video_files = []
    for ext in VALIDATION["supported_formats"]:
        video_files.extend(folder.glob(f"*{ext}"))
        video_files.extend(folder.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(video_files)} video files in {folder_path}")
    return sorted(video_files)

def validate_video(video_path):
    """
    Validate video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return False, f"File does not exist: {video_path}"
    
    # Check file extension
    if video_path.suffix.lower() not in VALIDATION["supported_formats"]:
        return False, f"Unsupported format: {video_path.suffix}"
    
    # Try to open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"Cannot open video: {video_path}"
    
    # Check duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    if duration < VALIDATION["min_video_duration"]:
        cap.release()
        return False, f"Video too short: {duration:.1f}s"
    
    if duration > VALIDATION["max_video_duration"]:
        cap.release()
        return False, f"Video too long: {duration:.1f}s"
    
    # Check resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_width, max_height = VALIDATION["max_resolution"]
    if width > max_width or height > max_height:
        cap.release()
        return False, f"Resolution too high: {width}x{height}"
    
    cap.release()
    return True, "Valid"

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        "path": str(video_path),
        "name": Path(video_path).name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    info["resolution"] = f"{info['width']}x{info['height']}"
    
    cap.release()
    return info
