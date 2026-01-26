# Complete Usage Guide - HCI Video Analyzer

## Table of Contents
1. [Installation](#installation)
2. [First Run](#first-run)
3. [Web Interface Guide](#web-interface-guide)
4. [CLI Guide](#cli-guide)
5. [Understanding Output](#understanding-output)
6. [Feature Descriptions](#feature-descriptions)
7. [Advanced Usage](#advanced-usage)
8. [Tips & Best Practices](#tips--best-practices)

---

## Installation

### Quick Install (Linux/macOS)
```bash
cd hci-video-analyzer
./install.sh
```

### Manual Install
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python test_installation.py
```

### Docker Install
```bash
docker build -t hci-analyzer .
docker run -p 8501:8501 -v $(pwd)/output:/app/output hci-analyzer
```

---

## First Run

### Test with Sample Video

1. Place a video file in a test folder
2. Run the analyzer:

```bash
# Web interface
streamlit run main.py

# CLI
python cli.py --input_folder ./test_videos --output test_results.json
```

3. Check output in `output/` directory

---

## Web Interface Guide

### Step 1: Launch Application
```bash
streamlit run main.py
```
Opens browser to `http://localhost:8501`

### Step 2: Configure Features (Sidebar)

**Feature Selection:**
- ✅ Check features you want to extract
- 💡 Tip: Start with 3-4 features for faster processing
- 🎯 Common combinations:
  - **UX Research**: Temporal, Spatial, Navigation, Cognitive Load
  - **Usability Testing**: Error/Recovery, Emotional/Affective, Success
  - **Complete Analysis**: Select All

**Quick Selection Buttons:**
- "Select All" - Enable all 16 features
- "Deselect All" - Clear all selections

**Similarity Detection:**
- ☑️ Enable/disable cross-video similarity
- 🎚️ Threshold slider (0.0 - 1.0)
  - 0.5 - Very loose matching
  - 0.75 - Balanced (recommended)
  - 0.9 - Very strict matching

### Step 3: Input Videos

**Option A: Folder Path**
1. Enter absolute or relative path
2. Example: `/home/user/videos` or `./my_videos`
3. Detects all supported formats (.mp4, .avi, .mov, .mkv, .webm)

**Option B: File Upload**
1. Click "Browse files"
2. Select multiple videos
3. Files uploaded to `temp/uploads/`

### Step 4: Start Analysis

1. Click **"🚀 Start Analysis"** button
2. **Progress Indicators:**
   - Overall progress bar
   - Per-video progress
   - Feature-by-feature status (✅/❌)
3. **Estimated Time:** ~1-2 minutes per minute of video

### Step 5: View Results

**Tab 1: 📈 Summary**
- Total videos processed
- Total features extracted
- Similar segments found
- Feature availability matrix

**Tab 2: 📹 Per-Video Results**
- Select video from dropdown
- View all extracted features as JSON
- Expandable categories

**Tab 3: 🔗 Similarities**
- List of similar segment groups
- Similarity scores
- Time intervals for each video
- Difference descriptions

**Tab 4: 📊 Visualizations**
- **Heatmaps**: Spatial interaction patterns
- **Timelines**: Temporal patterns over time
- **Comparison Charts**: Cross-video metrics
- Interactive Plotly charts

**Tab 5: 💾 Export**
- Download complete JSON results
- Generate text summary report
- Timestamp-named files

---

## CLI Guide

### Basic Commands

```bash
# Analyze all videos with all features
python cli.py --input_folder /path/to/videos

# Analyze with specific features
python cli.py --input_folder ./videos --features temporal,spatial,audio_verbal

# Enable similarity detection
python cli.py --input_folder ./videos --similarity --threshold 0.8

# Specify output file
python cli.py --input_folder ./videos --output my_results.json

# Verbose logging
python cli.py --input_folder ./videos --verbose
```

### List Available Features
```bash
python cli.py --list-features
```

### Example Workflows

**Quick Analysis (Fast)**
```bash
python cli.py \
  --input_folder ./videos \
  --features temporal,spatial,context
```

**Complete Analysis (Comprehensive)**
```bash
python cli.py \
  --input_folder ./videos \
  --features all \
  --similarity \
  --threshold 0.75 \
  --output complete_analysis.json \
  --verbose
```

**Batch Processing**
```bash
# Process multiple folders
for folder in study1 study2 study3; do
  python cli.py --input_folder ./$folder --output results_$folder.json
done
```

---

## Understanding Output

### JSON Structure

```json
{
  "analysis_metadata": {
    "timestamp": "ISO 8601 timestamp",
    "tool_version": "1.0.0",
    "total_videos": 3,
    "selected_features": ["temporal", "spatial", ...],
    "similarity_detection_enabled": true,
    "similarity_threshold": 0.75
  },
  
  "videos": [
    {
      "video_name": "filename.mp4",
      "video_path": "/full/path/to/video.mp4",
      "duration_seconds": 180.5,
      "resolution": "1920x1080",
      "features": {
        "temporal": { /* temporal metrics */ },
        "spatial": { /* spatial metrics */ },
        // ... other features
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
          "video_name": "video1.mp4",
          "time_interval": [45.2, 52.8],
          "features": { /* segment features */ }
        },
        {
          "video_name": "video2.mp4",
          "time_interval": [120.5, 131.2],
          "features": { /* segment features */ }
        }
      ],
      "differences": {
        "video1.mp4": "faster, fewer errors",
        "video2.mp4": "slower, more corrections"
      }
    }
  ],
  
  "comparative_analysis": {
    "metrics": {
      "temporal.total_interactions": {
        "mean": 42.5,
        "std": 8.3,
        "min": 30,
        "max": 55
      }
    }
  }
}
```

### Key Metrics Explained

**Temporal Features:**
- `total_interactions`: Number of detected user actions
- `average_interaction_duration`: Mean time per interaction (seconds)
- `average_pause_duration`: Mean pause between actions (seconds)
- `time_to_first_interaction`: Delay before first action (seconds)
- `interaction_rhythm`: "steady", "bursty", or "moderate"

**Spatial Features:**
- `cursor_trajectory_length_pixels`: Total cursor movement distance
- `average_cursor_speed_pixels_per_sec`: Mean cursor velocity
- `total_clicks`: Number of click events
- `heatmap_data`: 2D array of interaction density
- `total_scroll_distance_pixels`: Cumulative scroll amount

**Cognitive Load:**
- `average_decision_time`: Mean pause before actions (seconds)
- `action_frequency`: Actions per second
- `help_access_count`: Times user sought help
- `cognitive_load_score`: "low", "moderate", "high"

---

## Feature Descriptions

### 1. Temporal Features
**What it measures:** Timing and rhythm of interactions  
**Use for:** Task completion analysis, user pacing  
**Key metrics:** Interaction count, pauses, completion time

### 2. Spatial/Visual Features
**What it measures:** Cursor movement and screen coverage  
**Use for:** Navigation patterns, attention areas  
**Key metrics:** Trajectory, clicks, heatmap

### 3. Interaction Type
**What it measures:** Types of user actions  
**Use for:** Input modality analysis, gesture usage  
**Key metrics:** Clicks, drags, scrolls, keyboard input

### 4. Error and Recovery
**What it measures:** Mistakes and corrections  
**Use for:** Usability issues, interface problems  
**Key metrics:** Error count, recovery time, backtracking

### 5. Navigation
**What it measures:** Page/screen transitions  
**Use for:** Information architecture, flow analysis  
**Key metrics:** Path efficiency, backtracking, dead-ends

### 6. Content Interaction
**What it measures:** Text input and reading  
**Use for:** Form usability, content engagement  
**Key metrics:** Typing speed, corrections, reading time

### 7. Audio/Verbal
**What it measures:** Speech and vocal patterns  
**Use for:** Think-aloud protocols, sentiment  
**Key metrics:** Transcript, sentiment, hesitations

### 8. Facial/Body Language
**What it measures:** Non-verbal communication  
**Use for:** Emotional response, engagement  
**Key metrics:** Expressions, pose, gestures

### 9. UI State
**What it measures:** Interface elements present  
**Use for:** Context understanding, error detection  
**Key metrics:** Elements detected, error messages

### 10. Context
**What it measures:** Technical environment  
**Use for:** Platform comparisons, device effects  
**Key metrics:** Device type, OS, browser

### 11. Cognitive Load
**What it measures:** Mental effort indicators  
**Use for:** Interface complexity assessment  
**Key metrics:** Decision time, help-seeking, load score

### 12. Success/Outcome
**What it measures:** Task completion  
**Use for:** Effectiveness evaluation  
**Key metrics:** Completion rate, accuracy, efficiency

### 13. Emotional/Affective
**What it measures:** User emotions  
**Use for:** User experience quality  
**Key metrics:** Frustration, engagement, satisfaction

### 14. Collaboration
**What it measures:** Multi-user interaction  
**Use for:** Collaborative tools study  
**Key metrics:** Turn-taking, communication

### 15. Comparative
**What it measures:** Cross-video patterns  
**Use for:** User segmentation, outlier detection  
**Key metrics:** Variance, common paths, expertise

### 16. Technical/Performance
**What it measures:** System responsiveness  
**Use for:** Performance impact on UX  
**Key metrics:** Lag, loading times, frame rate

---

## Advanced Usage

### Custom Feature Extractor

See `examples/custom_extractor.py` for creating your own features.

### Batch Processing

```python
from examples.batch_analysis import batch_analyze_folders

folders = ["study1", "study2", "study3"]
batch_analyze_folders(folders, "output/batch_results")
```

### Convert to CSV

```bash
python examples/export_to_csv.py
```

### Programmatic Access

```python
from models.model_loader import ModelLoader
from feature_extractors.temporal import TemporalFeatureExtractor
import json

# Load models
loader = ModelLoader()
loader.load_all_models()

# Extract features
with TemporalFeatureExtractor("video.mp4", loader) as extractor:
    features = extractor.extract()

# Use the data
print(f"Interactions: {features['total_interactions']}")
print(f"Completion: {features['task_completion_time']:.1f}s")

# Save custom format
with open("custom_output.json", "w") as f:
    json.dump(features, f)
```

---

## Tips & Best Practices

### Performance Optimization

**1. Start Small**
- Test with 1-2 videos first
- Select 3-4 features initially
- Verify results before large batches

**2. Adjust Config**
Edit `config.py`:
```python
VIDEO_CONFIG = {
    "resize_height": 480,  # Lower = faster
    "skip_frames": 2,      # Process every 2nd frame
    "max_fps": 15          # Reduce FPS
}
```

**3. GPU Acceleration**
```python
MODELS = {
    "whisper": {"device": "cuda"},  # Use GPU
    "yolo": {"device": 0}           # GPU device 0
}
```

**4. Feature Selection**
Only extract what you need:
- Temporal + Spatial: ~30s per minute of video
- All 16 features: ~2-3 minutes per minute of video

### Quality Tips

**1. Video Quality**
- Higher resolution = better detection
- Good lighting for face/pose detection
- Clear audio for speech transcription

**2. Similarity Threshold**
- Start at 0.75 (balanced)
- Lower for diverse interactions
- Raise for very similar tasks

**3. Error Handling**
- Check `output/analyzer.log` for issues
- Use `--verbose` flag in CLI
- Validate videos before processing

### Common Use Cases

**UX Research Study**
```bash
python cli.py \
  --input_folder ./user_sessions \
  --features temporal,spatial,navigation,cognitive_load,success_outcome \
  --similarity --threshold 0.75
```

**A/B Test Comparison**
```bash
python cli.py --input_folder ./design_a --output results_a.json
python cli.py --input_folder ./design_b --output results_b.json
# Then compare results programmatically
```

**Accessibility Audit**
```bash
python cli.py \
  --input_folder ./sessions \
  --features interaction_type,context,cognitive_load,error_recovery
```

---

## Troubleshooting

See README.md section "Troubleshooting" for:
- Out of memory errors
- Slow processing
- Model download issues
- FFmpeg problems

---

**Questions?** Check the other documentation files:
- `README.md` - Complete reference
- `QUICKSTART.md` - Quick setup
- `PROJECT_SUMMARY.md` - Technical overview
- `VALIDATION_CHECKLIST.md` - Implementation details
