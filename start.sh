#!/bin/bash

# Quick Start Script for Kyutai TTS Service
# This script activates the virtual environment and starts the service

echo "🎵 Starting GPU-Optimized Kyutai TTS Service"
echo "====================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if main service file exists
if [ ! -f "run-tts-service.py" ]; then
    echo "❌ Service file not found: run-tts-service.py"
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
echo "🚀 Starting Kyutai TTS service..."
echo "📡 Service will be available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start the service
python3 run-tts-service.py 