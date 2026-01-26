#!/bin/bash
# Installation script for HCI Video Analyzer

set -e  # Exit on error

echo "================================================"
echo "HCI Video Analyzer - Installation Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10+ required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Check FFmpeg
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    else
        echo "❌ Please install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
fi
echo "✅ FFmpeg found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✅ Virtual environment created"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing Python dependencies..."
echo "(This may take several minutes...)"
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p output temp models/cached
echo "✅ Directories created"
echo ""

# Download NLTK data (for text processing)
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
echo "✅ NLTK data downloaded"
echo ""

echo "================================================"
echo "✅ Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the web interface:"
echo "   streamlit run main.py"
echo ""
echo "3. Or use the command line:"
echo "   python cli.py --input_folder /path/to/videos"
echo ""
echo "Models will be automatically downloaded on first run."
echo "Ensure you have ~5GB free disk space."
echo ""
