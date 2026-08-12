#!/bin/bash

# GPU-Optimized Kyutai TTS Service Installation Script (Modular Architecture)
# Sets up the Kyutai TTS service with the modern ROCm 7.x stack:
#   - Python 3.13 (required: AMD ROCm wheels are cp310-cp313 only)
#   - AMD-built PyTorch 2.9.1+rocm7.2.4 (native gfx1151 / Strix Halo support)
#   - moshi 0.2.13, modelscope, pydub, etc.
#   - uv package manager (no sudo needed for Python tooling)

set -e  # Exit on any error

echo "🎵 GPU-Optimized Kyutai TTS Service Installation (Modular)"
echo "========================================================"
echo ""
echo "This script will:"
echo "1. Check system requirements (Python, ROCm, GPU)"
echo "2. Install uv (fast Python package manager)"
echo "3. Create a Python 3.13 virtual environment (.venv)"
echo "4. Install AMD-built PyTorch with native gfx1151 support"
echo "5. Install all required dependencies"
echo "6. Verify the installation and modular architecture"
echo ""
echo "Tested on: Ubuntu 26.04, Python 3.13, AMD Strix Halo (gfx1151), ROCm 7.1.4"
echo "Architecture: Modular (main.py entry point)"
echo ""

# Check if Python is installed
echo "🔍 Step 1: Checking system requirements..."
echo "----------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Check ROCm
echo "🔍 Checking ROCm installation..."
if [ -d "/opt/rocm" ]; then
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    echo "✅ ROCm found: $ROCM_VER"
else
    echo "⚠️  ROCm not found in /opt/rocm. GPU acceleration will not work."
    echo "   Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html"
fi

# Check GPU
echo "🔍 Checking GPU..."
if command -v rocminfo &> /dev/null; then
    rocminfo 2>/dev/null | grep -E "Marketing Name|gfx" | head -4
else
    echo "⚠️  rocminfo not found. Cannot verify GPU."
fi

# Check modular architecture files
echo "🔍 Checking modular architecture files..."
MISSING_FILES=()

if [ ! -f "main.py" ]; then
    MISSING_FILES+=("main.py")
fi
if [ ! -f "config.py" ]; then
    MISSING_FILES+=("config.py")
fi
if [ ! -d "api" ]; then
    MISSING_FILES+=("api/")
fi
if [ ! -d "tts" ]; then
    MISSING_FILES+=("tts/")
fi
if [ ! -d "audio" ]; then
    MISSING_FILES+=("audio/")
fi
if [ ! -d "utils" ]; then
    MISSING_FILES+=("utils/")
fi

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing modular architecture components:"
    printf '   - %s\n' "${MISSING_FILES[@]}"
    echo ""
    echo "Please ensure you have the complete modular architecture."
    exit 1
else
    echo "✅ Modular architecture structure verified!"
fi

# Install uv
echo ""
echo "📦 Step 2: Installing uv..."
echo "----------------------------------------"
if command -v uv &> /dev/null; then
    echo "✅ uv already installed: $(uv --version)"
else
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ uv installed: $(uv --version)"
fi

# Create virtual environment with Python 3.13
echo ""
echo "📦 Step 3: Creating Python 3.13 virtual environment..."
echo "----------------------------------------"

if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment '.venv' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf .venv
        echo "✅ Existing virtual environment removed."
    else
        echo "ℹ️  Using existing virtual environment."
    fi
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creating new virtual environment with Python 3.13..."
    uv venv --python 3.13 .venv
    echo "✅ Virtual environment created successfully!"
fi

# Install AMD-built PyTorch stack
echo ""
echo "🚀 Step 4: Installing AMD-built PyTorch (native gfx1151 support)..."
echo "----------------------------------------"
echo "Note: Using AMD's ROCm wheels from repo.radeon.com because the official"
echo "PyTorch ROCm wheels do NOT include gfx1151 (Strix Halo) kernels."
echo "torch is pinned to 2.9.x because moshi requires torch<2.10."

VIRTUAL_ENV=.venv uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchaudio-2.9.0%2Brocm7.2.4.gite3c6ee2b-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchvision-0.24.0%2Brocm7.2.4.gitb919bd0c-cp313-cp313-linux_x86_64.whl

if [ $? -eq 0 ]; then
    echo "✅ AMD PyTorch stack installed successfully!"
else
    echo "❌ Failed to install AMD PyTorch stack."
    exit 1
fi

# Install other dependencies
echo ""
echo "📚 Step 5: Installing dependencies..."
echo "----------------------------------------"
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies."
    exit 1
fi

# Install ffmpeg (static build, no sudo needed) for pydub MP3 export
echo ""
echo "🎬 Step 6: Installing ffmpeg (static build)..."
echo "----------------------------------------"
if command -v ffmpeg &> /dev/null; then
    echo "✅ ffmpeg already available: $(ffmpeg -version 2>&1 | head -1)"
elif [ -f "$HOME/bin/ffmpeg" ]; then
    echo "✅ ffmpeg already in ~/bin"
else
    echo "📦 Downloading static ffmpeg..."
    mkdir -p "$HOME/bin"
    # Use a subshell so the working directory is preserved for later steps
    (
        cd /tmp
        curl -sL -o ffmpeg.tar.xz "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        tar xf ffmpeg.tar.xz
        FF_DIR=$(ls -d ffmpeg-*-static | head -1)
        cp "$FF_DIR/ffmpeg" "$FF_DIR/ffprobe" "$HOME/bin/"
    )
    chmod +x "$HOME/bin/ffmpeg" "$HOME/bin/ffprobe"
    echo "✅ ffmpeg installed to ~/bin"
fi

# Verify installation
echo ""
echo "✅ Step 7: Verifying installation..."
echo "----------------------------------------"

# Verify GPU works
echo "🔍 Verifying GPU..."
LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib" .venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'HIP: {torch.version.hip}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(1024, 1024, device='cuda')
    print(f'GPU matmul OK: {round((x @ x).sum().item(), 2)}')
else:
    print('⚠️  GPU not available - service will run on CPU (slow)')
"

# Verify modular architecture imports
echo "🏗️  Verifying modular architecture..."
LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib" .venv/bin/python -c "
import config
from utils.ssml_parser import parse_ssml
from audio.processing import initialize_zipenhancer
from tts.engine import initialize_environment
from api.models import TTSRequest
from api.routes import router
from main import main
print('✅ All modular components imported successfully!')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "🏗️  Modular Architecture Summary:"
echo "   📦 Entry Point: main.py"
echo "   ⚙️  Configuration: config.py"
echo "   🌐 API Layer: api/ (models.py, routes.py)"
echo "   🎤 TTS Engine: tts/engine.py"
echo "   🎵 Audio Processing: audio/processing.py"
echo "   🛠️  Utilities: utils/ssml_parser.py"
echo ""
echo "🚀 To start the service:"
echo "   ./start.sh"
echo "   OR: source .venv/bin/activate && python main.py"
echo ""
echo "📚 For more information, see README.md"
echo ""

# Ask if user wants to start the service now
read -p "Do you want to start the Kyutai TTS service now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Kyutai TTS service (modular architecture)..."
    echo "📡 Service will be available at: http://localhost:7861"
    echo "Press Ctrl+C to stop the service"
    echo ""
    ./start.sh
else
    echo "ℹ️  You can start the service later with: ./start.sh"
fi
