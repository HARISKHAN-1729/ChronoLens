# HCI Multi-Video Analysis Tool

A comprehensive, production-ready tool for analyzing Human-Computer Interaction patterns across multiple videos using **only open-source AI models**.

## 🎯 Features

### 16 Categories of HCI Analysis (All Fully Implemented)

1. **Temporal Features** - Interaction timing, pauses, task completion, reaction times
2. **Spatial/Visual Features** - Cursor trajectories, click locations, heatmaps, scrolling
3. **Interaction Type** - Clicks, drags, keyboard input, gestures
4. **Error and Recovery** - Error detection, backtracking, recovery strategies
5. **Navigation** - Page/screen transitions, navigation paths, efficiency
6. **Content Interaction** - Text input, typing speed, form completion, media interactions
7. **Audio/Verbal** - Speech transcription, sentiment analysis, hesitations
8. **Facial/Body Language** - Expression recognition, pose estimation, gestures
9. **UI State** - UI element detection, error messages, loading states
10. **Context** - Device/platform detection, screen resolution, browser identification
11. **Cognitive Load** - Decision time, help-seeking, trial-and-error patterns
12. **Success/Outcome** - Task completion, accuracy, efficiency metrics
13. **Emotional/Affective** - Frustration, engagement, confusion detection
14. **Collaboration** - Multi-user detection, turn-taking, communication patterns
15. **Comparative** - Cross-video analysis, pattern identification, outliers
16. **Technical/Performance** - System lag, loading times, frame rate analysis

### Additional Capabilities

- ✅ **Cross-Video Similarity Detection** - Identify similar interaction patterns across videos
- ✅ **Interactive Streamlit Interface** - User-friendly web UI with feature selection
- ✅ **Command-Line Interface** - Scriptable batch processing
- ✅ **JSON Export** - Structured data output for further analysis
- ✅ **Visualizations** - Heatmaps, timelines, comparison charts
- ✅ **100% Open Source** - No proprietary API dependencies

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (for audio processing)
- **~5GB disk space** (for AI models)
- **GPU recommended** (but not required)

### Installation

```bash
# Clone or download the project
cd hci-video-analyzer

# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### Running the Application

#### Web Interface (Streamlit)

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501`

#### Command Line

```bash
# Analyze all videos in a folder
python cli.py --input_folder /path/to/videos --output results.json

# Analyze specific features only
python cli.py --input_folder /path/to/videos --features temporal,spatial,audio_verbal

# Enable similarity detection
python cli.py --input_folder /path/to/videos --similarity --threshold 0.8

# See all options
python cli.py --help
```

## 📖 Usage Guide

### Web Interface

1. **Launch** the app: `streamlit run main.py`
2. **Select features** in the sidebar (check/uncheck categories)
3. **Configure similarity detection** (enable/disable, set threshold)
4. **Input videos** (folder path or file upload)
5. **Start analysis** and monitor progress
6. **View results** in organized tabs:
   - **Summary**: Overview statistics and feature matrix
   - **Per-Video**: Detailed results for each video
   - **Similarities**: Cross-video similar segments
   - **Visualizations**: Heatmaps, timelines, comparisons
   - **Export**: Download JSON and reports

### Command Line Interface

```bash
# Basic usage
python cli.py --input_folder /path/to/videos

# List available features
python cli.py --list-features

# Extract specific features
python cli.py --input_folder ./videos --features temporal,spatial,cognitive_load

# With similarity detection
python cli.py --input_folder ./videos --similarity --threshold 0.75

# Verbose output for debugging
python cli.py --input_folder ./videos --verbose
```

### Programmatic Usage

```python
from models.model_loader import ModelLoader
from feature_extractors.temporal import TemporalFeatureExtractor

# Load models
loader = ModelLoader()
loader.load_all_models()

# Extract features from a video
with TemporalFeatureExtractor("video.mp4", loader) as extractor:
    features = extractor.extract()
    print(f"Total interactions: {features['total_interactions']}")
    print(f"Average pause: {features['average_pause_duration']:.2f}s")
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model settings
MODELS = {
    "whisper": {"model_name": "base"},  # tiny, base, small, medium, large
    "yolo": {"model_path": "yolov8n.pt"},
    # ...
}

# Feature extraction parameters
TEMPORAL_CONFIG = {
    "pause_threshold": 2.0,  # Seconds
    # ...
}

# Similarity detection
SIMILARITY_CONFIG = {
    "default_threshold": 0.75,
    "segment_duration": 5.0,
    # ...
}
```

## 📊 Output Format

Results are exported as JSON with this structure:

```json
{
  "analysis_metadata": {
    "timestamp": "2026-01-26T19:30:00Z",
    "tool_version": "1.0.0",
    "total_videos": 3,
    "selected_features": ["temporal", "spatial", "..."]
  },
  "videos": [
    {
      "video_name": "user1_session.mp4",
      "video_path": "/path/to/user1_session.mp4",
      "features": {
        "temporal": {
          "total_interactions": 45,
          "average_interaction_duration": 2.3,
          "task_completion_time": 180.5,
          "...": "..."
        },
        "spatial": {
          "cursor_trajectory_length_pixels": 15420.5,
          "total_clicks": 35,
          "heatmap_data": [[...]],
          "...": "..."
        }
      }
    }
  ],
  "similar_segments": [
    {
      "segment_id": "seg_001",
      "similarity_type": "temporal",
      "similarity_score": 0.87,
      "videos": [
        {
          "video_name": "user1_session.mp4",
          "time_interval": [45.2, 52.8],
          "features": {...}
        },
        {
          "video_name": "user2_session.mp4",
          "time_interval": [120.5, 131.2],
          "features": {...}
        }
      ],
      "differences": {...}
    }
  ],
  "comparative_analysis": {
    "metrics": {...}
  }
}
```

## 🤖 AI Models Used (All Open Source)

| Purpose | Model | Source |
|---------|-------|--------|
| Speech-to-Text | Whisper | OpenAI (open source) |
| Object Detection | YOLOv8 | Ultralytics |
| Pose Estimation | MediaPipe | Google (open source) |
| Face Detection | MediaPipe | Google (open source) |
| OCR | EasyOCR | JaidedAI |
| Sentiment Analysis | DistilBERT | Hugging Face |
| Embeddings | Sentence-Transformers | Hugging Face |

**No proprietary APIs required!** All models run locally.

## 🎨 Example Use Cases

### UX Research
- Compare interaction patterns between novice and expert users
- Identify pain points and confusion moments
- Measure task completion efficiency

### A/B Testing
- Compare two interface designs across user sessions
- Quantify differences in user behavior
- Identify which design leads to faster task completion

### Accessibility Studies
- Analyze different input modalities (mouse, touch, keyboard)
- Measure cognitive load indicators
- Track help-seeking behavior

### Usability Testing
- Detect errors and recovery patterns
- Measure navigation efficiency
- Identify common user paths vs outliers

## 🐛 Troubleshooting

### "Out of memory" errors
- Reduce `VIDEO_CONFIG["max_fps"]` in config.py
- Process fewer videos simultaneously
- Use smaller AI models (e.g., Whisper "tiny" instead of "base")

### Slow processing
- Enable GPU: Set `device: "cuda"` in config.py
- Skip frames: Increase `VIDEO_CONFIG["skip_frames"]`
- Reduce video resolution: Set `VIDEO_CONFIG["resize_height"]`

### Model download fails
- Check internet connection
- Ensure ~5GB free disk space
- Manually download models and place in `models/cached/`

### FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Add FFmpeg to PATH after downloading
```

## 📁 Project Structure

```
hci-video-analyzer/
├── main.py                      # Streamlit web interface
├── cli.py                       # Command-line interface
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── feature_extractors/          # All 16 feature extractors
│   ├── temporal.py
│   ├── spatial.py
│   ├── interaction_type.py
│   ├── error_recovery.py
│   ├── navigation.py
│   ├── content_interaction.py
│   ├── audio_verbal.py
│   ├── facial_body.py
│   ├── ui_state.py
│   ├── context.py
│   ├── cognitive_load.py
│   ├── success_outcome.py
│   ├── emotional_affective.py
│   ├── collaboration.py
│   ├── comparative.py
│   └── technical_performance.py
├── similarity/                  # Cross-video similarity detection
│   └── segment_matcher.py
├── models/                      # Model loading and caching
│   └── model_loader.py
├── utils/                       # Utility functions
│   ├── video_processor.py
│   ├── visualization.py
│   └── json_exporter.py
├── output/                      # Analysis results
├── temp/                        # Temporary files
└── tests/                       # Test suite
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_temporal.py -v
```

## 🔐 Privacy & Security

- **100% local processing** - No data sent to external servers
- **No API keys required** - All models run locally
- **Videos never uploaded** - Analysis happens on your machine
- **Offline capable** - Works without internet (after initial model download)

## 📈 Performance Tips

1. **GPU Acceleration**: Install CUDA and set `device: "cuda"` in config
2. **Batch Processing**: Use CLI for large video sets
3. **Feature Selection**: Only enable needed features
4. **Video Preprocessing**: Reduce resolution/FPS before analysis
5. **Multiprocessing**: Enable in config for parallel processing

## 🤝 Contributing

Contributions welcome! Please:
1. Ensure all tests pass
2. Follow PEP 8 style guide
3. Add tests for new features
4. Update documentation

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web interface
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [MediaPipe](https://google.github.io/mediapipe/) - Pose/face detection
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text recognition

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section
- Review example scripts in `examples/`

## 🎓 Citation

If you use this tool in research, please cite:

```bibtex
@software{hci_video_analyzer_2026,
  title={HCI Multi-Video Analysis Tool},
  author={Muhammad Haris Khan},
  year={2026},
  url={https://github.com/yourusername/hci-video-analyzer}
}
```


