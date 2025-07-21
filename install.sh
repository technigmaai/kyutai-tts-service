#!/bin/bash

# GPU-Optimized TTS Service Installation Script
# This script sets up the TTS service with all dependencies

set -e  # Exit on any error

echo "🎵 GPU-Optimized TTS Service Installation"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Check if CUDA is available
echo "🔍 Checking CUDA availability..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✅ CUDA is available"
else
    echo "⚠️  CUDA not available. Installing CPU-only PyTorch..."
    CUDA_INDEX=""
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Install PyTorch with ROCm support (as tested)
echo "🚀 Installing PyTorch with ROCm support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import fastapi
import uvicorn
import pydub
import librosa
import psutil
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "🚀 To start the service:"
echo "   source .venv/bin/activate"
echo "   python run-tts-service.py"
echo ""
echo "📚 For more information, see README.md"
echo "" 