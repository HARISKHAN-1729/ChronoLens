"""
Configuration file for HCI Video Analyzer
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models" / "cached"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS = {
    "whisper": {
        "model_name": "base",  # Options: tiny, base, small, medium, large
        "device": "cpu"  # Change to "cuda" if GPU available
    },
    "yolo": {
        "model_path": "yolov8n.pt",
        "confidence": 0.5
    },
    "sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2"
    },
    "face_recognition": {
        "backend": "opencv",
        "detector_backend": "opencv"
    },
    "pose_estimation": {
        "model_complexity": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5
    },
    "ocr": {
        "languages": ["en"],
        "gpu": False
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

# Feature extraction parameters
TEMPORAL_CONFIG = {
    "interaction_threshold": 0.01,
    "pause_threshold": 2.0,
    "min_interaction_duration": 0.1
}

SPATIAL_CONFIG = {
    "grid_size": (10, 10),
    "cursor_detection_sensitivity": 5,
    "trajectory_smoothing": 3
}

INTERACTION_CONFIG = {
    "double_click_threshold": 0.3,
    "drag_threshold": 10,
    "scroll_threshold": 5
}

ERROR_CONFIG = {
    "max_error_recovery_time": 10.0,
    "repeated_error_window": 30.0
}

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_duration": 30,
    "hesitation_words": ["um", "uh", "er", "hmm", "ah", "like"],
    "sentiment_window": 5.0
}

COGNITIVE_LOAD_CONFIG = {
    "decision_threshold": 3.0,
    "high_load_threshold": 5.0,
    "action_frequency_window": 10.0
}

# Similarity detection parameters
SIMILARITY_CONFIG = {
    "segment_duration": 5.0,
    "overlap": 0.5,
    "default_threshold": 0.75,
    "dtw_window": 10,
    "min_segment_length": 2.0
}

# Video processing parameters
VIDEO_CONFIG = {
    "max_fps": 30,
    "resize_height": 720,
    "skip_frames": 1,
    "batch_size": 32
}

# UI Configuration
UI_CONFIG = {
    "page_title": "HCI Video Analyzer",
    "page_icon": "🎥",
    "layout": "wide",
    "max_upload_size": 500
}

# Visualization parameters
VIZ_CONFIG = {
    "heatmap_colormap": "hot",
    "timeline_height": 400,
    "comparison_chart_height": 500,
    "dpi": 100
}

# Export settings
EXPORT_CONFIG = {
    "json_indent": 2,
    "include_raw_data": False,
    "compress_output": False,
    "save_visualizations": True
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "analyzer.log"
}

# Performance settings
PERFORMANCE = {
    "use_multiprocessing": True,
    "max_workers": 4,
    "gpu_memory_fraction": 0.8,
    "cache_features": True
}

# Validation
VALIDATION = {
    "max_video_duration": 3600,
    "min_video_duration": 1,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "max_resolution": (3840, 2160)
}
