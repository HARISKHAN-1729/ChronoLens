"""
JSON export utilities
"""
import json
from pathlib import Path
from datetime import datetime
import logging
from config import EXPORT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_results(all_results, similar_segments, output_path, metadata=None):
    """
    Export analysis results to JSON
    
    Args:
        all_results: Dict of video results
        similar_segments: List of similar segment groups
        output_path: Output file path
        metadata: Additional metadata to include
    """
    output = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "tool_version": "1.0.0",
            "total_videos": len(all_results),
            **(metadata or {})
        },
        "videos": [],
        "similar_segments": similar_segments
    }
    
    # Format video results
    for video_path, features in all_results.items():
        video_data = {
            "video_name": Path(video_path).name,
            "video_path": str(video_path),
            "features": features
        }
        
        # Add basic video info if available
        if "context" in features and features["context"]:
            video_data["duration_seconds"] = features["context"].get("video_duration", 0)
            video_data["resolution"] = features["context"].get("resolution", "unknown")
        
        output["videos"].append(video_data)
    
    # Add comparative analysis
    if len(all_results) > 1:
        output["comparative_analysis"] = generate_comparative_analysis(all_results)
    
    # Write JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=EXPORT_CONFIG["json_indent"])
    
    logger.info(f"Results exported to: {output_path}")
    return output_path

def generate_comparative_analysis(all_results):
    """Generate comparative statistics across all videos"""
    comparative = {
        "total_videos": len(all_results),
        "metrics": {}
    }
    
    # Collect metrics across videos
    all_metrics = {}
    
    for video_path, features in all_results.items():
        for category, category_features in features.items():
            if not isinstance(category_features, dict):
                continue
            
            for metric, value in category_features.items():
                if isinstance(value, (int, float)):
                    key = f"{category}.{metric}"
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
    
    # Calculate statistics
    import numpy as np
    
    for metric_key, values in all_metrics.items():
        if len(values) > 0:
            comparative["metrics"][metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    return comparative
