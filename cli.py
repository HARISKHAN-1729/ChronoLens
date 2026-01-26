#!/usr/bin/env python3
"""
Command-line interface for HCI Video Analyzer
"""
import argparse
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model_loader import ModelLoader
from utils.video_processor import get_video_files, validate_video
from utils.json_exporter import export_results
from similarity.segment_matcher import SegmentMatcher

# Import all feature extractors
from feature_extractors.temporal import TemporalFeatureExtractor
from feature_extractors.spatial import SpatialFeatureExtractor
from feature_extractors.interaction_type import InteractionTypeExtractor
from feature_extractors.error_recovery import ErrorRecoveryExtractor
from feature_extractors.navigation import NavigationExtractor
from feature_extractors.content_interaction import ContentInteractionExtractor
from feature_extractors.audio_verbal import AudioVerbalExtractor
from feature_extractors.facial_body import FacialBodyExtractor
from feature_extractors.ui_state import UIStateExtractor
from feature_extractors.context import ContextExtractor
from feature_extractors.cognitive_load import CognitiveLoadExtractor
from feature_extractors.success_outcome import SuccessOutcomeExtractor
from feature_extractors.emotional_affective import EmotionalAffectiveExtractor
from feature_extractors.collaboration import CollaborationExtractor
from feature_extractors.comparative import ComparativeExtractor
from feature_extractors.technical_performance import TechnicalPerformanceExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_EXTRACTORS = {
    "temporal": TemporalFeatureExtractor,
    "spatial": SpatialFeatureExtractor,
    "interaction_type": InteractionTypeExtractor,
    "error_recovery": ErrorRecoveryExtractor,
    "navigation": NavigationExtractor,
    "content_interaction": ContentInteractionExtractor,
    "audio_verbal": AudioVerbalExtractor,
    "facial_body": FacialBodyExtractor,
    "ui_state": UIStateExtractor,
    "context": ContextExtractor,
    "cognitive_load": CognitiveLoadExtractor,
    "success_outcome": SuccessOutcomeExtractor,
    "emotional_affective": EmotionalAffectiveExtractor,
    "collaboration": CollaborationExtractor,
    "comparative": ComparativeExtractor,
    "technical_performance": TechnicalPerformanceExtractor
}

def main():
    parser = argparse.ArgumentParser(
        description="HCI Video Analysis Tool - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all videos in a folder with all features
  %(prog)s --input_folder /path/to/videos --output results.json
  
  # Analyze specific features only
  %(prog)s --input_folder /path/to/videos --features temporal,spatial,audio_verbal
  
  # Enable similarity detection with custom threshold
  %(prog)s --input_folder /path/to/videos --similarity --threshold 0.8
  
  # Verbose output
  %(prog)s --input_folder /path/to/videos --verbose
        """
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing videos to analyze"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: output/analysis_TIMESTAMP.json)"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="Comma-separated feature names or 'all' (default: all)"
    )
    
    parser.add_argument(
        "--similarity",
        action="store_true",
        help="Enable similarity detection"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold 0-1 (default: 0.75)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all available features and exit"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List features and exit
    if args.list_features:
        print("Available features:")
        for key in FEATURE_EXTRACTORS.keys():
            print(f"  - {key}")
        return 0
    
    # Determine output path
    if args.output is None:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Get video files
    logger.info(f"Scanning folder: {args.input_folder}")
    try:
        video_files = get_video_files(args.input_folder)
    except Exception as e:
        logger.error(f"Error accessing folder: {e}")
        return 1
    
    if not video_files:
        logger.error("No video files found")
        return 1
    
    logger.info(f"Found {len(video_files)} video(s)")
    
    # Determine which features to extract
    if args.features.lower() == "all":
        selected_features = list(FEATURE_EXTRACTORS.keys())
    else:
        selected_features = [f.strip() for f in args.features.split(",")]
        
        # Validate feature names
        invalid = [f for f in selected_features if f not in FEATURE_EXTRACTORS]
        if invalid:
            logger.error(f"Invalid feature names: {invalid}")
            logger.info("Use --list-features to see available features")
            return 1
    
    logger.info(f"Will extract {len(selected_features)} feature(s)")
    
    # Load models
    logger.info("Loading AI models...")
    try:
        model_loader = ModelLoader()
        model_loader.load_all_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return 1
    
    # Process videos
    all_results = {}
    
    for idx, video_path in enumerate(video_files):
        logger.info(f"Processing [{idx+1}/{len(video_files)}]: {video_path.name}")
        
        # Validate
        is_valid, message = validate_video(video_path)
        if not is_valid:
            logger.warning(f"Skipping {video_path.name}: {message}")
            continue
        
        video_results = {}
        
        # Extract features
        for feature_key in selected_features:
            extractor_class = FEATURE_EXTRACTORS[feature_key]
            
            try:
                logger.info(f"  Extracting {feature_key}...")
                
                with extractor_class(video_path, model_loader) as extractor:
                    features = extractor.extract()
                    video_results[feature_key] = features
                
                logger.info(f"  ✓ {feature_key} complete")
                
            except Exception as e:
                logger.error(f"  ✗ Error in {feature_key}: {e}")
                if args.verbose:
                    logger.exception("Detailed error:")
                video_results[feature_key] = None
        
        all_results[str(video_path)] = video_results
    
    logger.info("Video processing complete")
    
    # Similarity detection
    similar_segments = []
    if args.similarity and len(all_results) > 1:
        logger.info("Detecting similar segments...")
        try:
            matcher = SegmentMatcher(all_results, args.threshold)
            similar_segments = matcher.find_all_similarities(selected_features)
            logger.info(f"Found {len(similar_segments)} similar segment groups")
        except Exception as e:
            logger.error(f"Error in similarity detection: {e}")
            if args.verbose:
                logger.exception("Detailed error:")
    
    # Export results
    logger.info(f"Exporting results to: {args.output}")
    try:
        export_results(
            all_results,
            similar_segments,
            args.output,
            metadata={
                "selected_features": selected_features,
                "similarity_detection_enabled": args.similarity,
                "similarity_threshold": args.threshold
            }
        )
        logger.info("✓ Export complete")
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return 1
    
    logger.info("Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
