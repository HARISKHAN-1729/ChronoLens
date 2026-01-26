#!/usr/bin/env python3
"""
Simple test to verify installation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import MODELS
        print("✅ Config imported")
        
        from models.model_loader import ModelLoader
        print("✅ ModelLoader imported")
        
        from feature_extractors.temporal import TemporalFeatureExtractor
        print("✅ TemporalFeatureExtractor imported")
        
        from feature_extractors.spatial import SpatialFeatureExtractor
        print("✅ SpatialFeatureExtractor imported")
        
        from similarity.segment_matcher import SegmentMatcher
        print("✅ SegmentMatcher imported")
        
        from utils.video_processor import get_video_files
        print("✅ Utils imported")
        
        print("\n✅ All core modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available"""
    print("\nTesting dependencies...")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import whisper
        print("✅ Whisper available")
        
        import mediapipe
        print("✅ MediaPipe available")
        
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
        
        print("\n✅ All dependencies available!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dependency check failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("HCI Video Analyzer - Installation Test")
    print("=" * 50)
    print()
    
    imports_ok = test_imports()
    deps_ok = test_dependencies()
    
    print()
    print("=" * 50)
    
    if imports_ok and deps_ok:
        print("✅ Installation test passed!")
        print()
        print("Ready to use! Try:")
        print("  streamlit run main.py")
        print("  python cli.py --list-features")
        sys.exit(0)
    else:
        print("❌ Installation test failed")
        print("Please run: ./install.sh")
        sys.exit(1)
