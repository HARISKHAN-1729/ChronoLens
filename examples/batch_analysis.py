#!/usr/bin/env python3
"""
Example: Batch process multiple video folders
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader
from utils.video_processor import get_video_files
from utils.json_exporter import export_results
from feature_extractors.temporal import TemporalFeatureExtractor
from feature_extractors.spatial import SpatialFeatureExtractor

def batch_analyze_folders(folder_list, output_dir):
    """Analyze videos from multiple folders"""
    
    # Load models once
    print("Loading AI models...")
    model_loader = ModelLoader()
    model_loader.load_all_models()
    
    for folder_path in folder_list:
        print(f"\nProcessing folder: {folder_path}")
        
        # Get videos
        video_files = get_video_files(folder_path)
        print(f"Found {len(video_files)} videos")
        
        all_results = {}
        
        for video_path in video_files:
            print(f"  Analyzing: {video_path.name}")
            
            video_results = {}
            
            # Extract temporal features
            with TemporalFeatureExtractor(video_path, model_loader) as extractor:
                video_results["temporal"] = extractor.extract()
            
            # Extract spatial features
            with SpatialFeatureExtractor(video_path, model_loader) as extractor:
                video_results["spatial"] = extractor.extract()
            
            all_results[str(video_path)] = video_results
        
        # Export results for this folder
        folder_name = Path(folder_path).name
        output_file = Path(output_dir) / f"results_{folder_name}.json"
        export_results(all_results, [], output_file)
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    folders = [
        "/path/to/folder1",
        "/path/to/folder2",
        "/path/to/folder3"
    ]
    
    output_directory = "output/batch_results"
    
    batch_analyze_folders(folders, output_directory)
    print("\nBatch processing complete!")
