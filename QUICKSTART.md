# Quick Start Guide

## Installation (5 minutes)

### Option 1: Automated Installation (Linux/macOS)

```bash
cd ChronoLens
./install.sh
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

## First Run (Web Interface)

```bash
# Activate environment
source venv/bin/activate

# Run Streamlit app
streamlit run main.py
```

Your browser will open to `http://localhost:8501`

## First Run (Command Line)

```bash
# Activate environment
source venv/bin/activate

# Analyze videos
python cli.py --input_folder /path/to/your/videos
```

Results saved to `output/analysis_TIMESTAMP.json`

## Using the Web Interface

1. **Select Features** (sidebar)
   - Check the features you want to extract
   - Use "Select All" or pick specific categories

2. **Configure Similarity** (sidebar)
   - Enable/disable similarity detection
   - Adjust threshold (0.0 = very loose, 1.0 = exact match)

3. **Input Videos**
   - **Folder Path**: Enter path to folder containing videos
   - **File Upload**: Upload video files directly

4. **Start Analysis**
   - Click "Start Analysis"
   - Monitor progress in real-time

5. **View Results**
   - **Summary**: Overview statistics
   - **Per-Video**: Detailed results for each video
   - **Similarities**: Cross-video matching segments
   - **Visualizations**: Heatmaps and charts
   - **Export**: Download JSON results

## Example: Analyze Videos with Specific Features

```bash
python cli.py \
  --input_folder ./test_videos \
  --features temporal,spatial,audio_verbal \
  --output results.json \
  --verbose
```

## Example: Enable Similarity Detection

```bash
python cli.py \
  --input_folder ./test_videos \
  --similarity \
  --threshold 0.8 \
  --output results_with_similarity.json
```

## Understanding Results

### JSON Structure

```json
{
  "analysis_metadata": {
    "timestamp": "2026-01-26T19:30:00Z",
    "total_videos": 3
  },
  "videos": [
    {
      "video_name": "session1.mp4",
      "features": {
        "temporal": { "total_interactions": 45, ... },
        "spatial": { "total_clicks": 35, ... }
      }
    }
  ],
  "similar_segments": [...]
}
```

### Key Metrics

- **total_interactions**: Number of detected user actions
- **average_pause_duration**: Average time between actions
- **cursor_trajectory_length**: Total cursor movement distance
- **total_clicks**: Number of click events
- **typing_speed_wpm**: Words per minute for text input
- **sentiment_score**: Overall emotional tone (-1 to +1)
- **cognitive_load_score**: Estimated mental effort

## Troubleshooting

### Models not downloading
- Check internet connection
- Ensure ~5GB free disk space
- Models cache to `models/cached/`

### Out of memory
- Edit `config.py`:
  - Set `VIDEO_CONFIG["resize_height"] = 480`
  - Set `VIDEO_CONFIG["skip_frames"] = 2`

### Slow processing
- Use GPU: Set `MODELS["whisper"]["device"] = "cuda"`
- Process fewer features
- Reduce video resolution

## What Features Should I Use?

### For UX Research
- ✅ Temporal, Spatial, Navigation, Cognitive Load, Success/Outcome

### For Usability Testing
- ✅ Error/Recovery, Navigation, Emotional/Affective, Success/Outcome

### For A/B Testing
- ✅ Temporal, Spatial, Navigation, Success/Outcome, Comparative

### For Accessibility Studies
- ✅ Interaction Type, Cognitive Load, Error/Recovery, Context

### For Complete Analysis
- ✅ Select All (16 features)

## Need Help?

- Check `README.md` for detailed documentation
- Review `examples/` directory for code samples
- See `VALIDATION_CHECKLIST.md` for implementation details

## Performance Tips

1. **Start Small**: Test with 1-2 videos first
2. **Select Needed Features**: Don't extract everything if you don't need it
3. **Use CLI for Batches**: CLI is faster for large video sets
4. **Enable GPU**: Significantly faster if you have CUDA-capable GPU
5. **Preprocess Videos**: Reduce resolution/FPS before analysis

Enjoy analyzing! 🎥
