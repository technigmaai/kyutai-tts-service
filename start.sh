#!/bin/bash

# Quick Start Script for Kyutai TTS Service (Modular Architecture)
# This script activates the virtual environment and starts the service

echo "🎵 Starting GPU-Optimized Kyutai TTS Service (Modular)"
echo "==============================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if main service file exists
if [ ! -f "main.py" ]; then
    echo "❌ Service entry point not found: main.py"
    echo "💡 Make sure you're in the correct directory with the modular architecture."
    exit 1
fi

# Check if config file exists
if [ ! -f "config.py" ]; then
    echo "❌ Configuration file not found: config.py"
    echo "💡 The modular architecture requires config.py to be present."
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  No GPU detected. Service will run on CPU (slower).')
"

echo ""
echo "🏗️  Modular Architecture Components:"
echo "   📦 Main Entry Point: main.py"
echo "   ⚙️  Configuration: config.py"
echo "   🌐 API Layer: api/"
echo "   🎤 TTS Engine: tts/"
echo "   🎵 Audio Processing: audio/"
echo "   🛠️  Utilities: utils/"
echo ""
echo "🚀 Starting Kyutai TTS service..."
echo "📡 Service will be available at: http://localhost:7861"
echo "📚 API documentation: http://localhost:7861/docs"
echo "🔧 ZipEnhancer status: http://localhost:7861/api/zipenhancer/status"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start the service with the new modular entry point
python3 main.py 