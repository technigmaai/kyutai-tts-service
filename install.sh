#!/bin/bash

# GPU-Optimized Kyutai TTS Service Installation Script (Modular Architecture)
# This script sets up the Kyutai TTS service with all dependencies including ROCm

set -e  # Exit on any error

echo "🎵 GPU-Optimized Kyutai TTS Service Installation (Modular)"
echo "========================================================"
echo ""
echo "This script will:"
echo "1. Check system requirements (Python, existing venv)"
echo "2. Optionally install ROCm drivers (for AMD GPU acceleration)"
echo "3. Create a Python virtual environment (.venv)"
echo "4. Check GPU support (ROCm for AMD GPUs)"
echo "5. Install PyTorch with ROCm support"
echo "6. Install all required dependencies"
echo "7. Verify the installation and modular architecture"
echo "8. Optionally start the service"
echo ""
echo "Tested on: Ubuntu 24.04, Python 3.12.10, AMD Strix Halo GPU"
echo "Architecture: Modular (main.py entry point)"
echo ""

# Check if Python is installed
echo "🔍 Step 1: Checking system requirements..."
echo "----------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Check if modular architecture files exist
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

# Check if ROCm is installed
echo "🔍 Checking ROCm installation..."
if command -v rocminfo &> /dev/null; then
    echo "✅ ROCm is already installed!"
    rocminfo
else
    echo "⚠️  ROCm is not installed. This is required for AMD GPU acceleration."
    read -p "Do you want to install ROCm now? (requires sudo) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔧 Step 2: Installing ROCm drivers..."
        echo "----------------------------------------"
        
        # Check if running as root
        if [ "$EUID" -ne 0 ]; then
            echo "⚠️  ROCm installation requires root privileges."
            echo "Please run: sudo ./install.sh"
            exit 1
        fi
        
        echo "📦 Installing ROCm..."
        
        # Add ROCm repository
        echo "Adding ROCm repository..."
        wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | tee /usr/share/keyrings/rocm-keyring.gpg > /dev/null
        echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/debian jammy main' | tee /etc/apt/sources.list.d/rocm.list
        
        # Update package list
        apt update
        
        # Install ROCm
        echo "Installing ROCm packages..."
        apt install -y rocm-hip-sdk
        
        # Install additional ROCm packages
        echo "Installing additional ROCm packages..."
        apt install -y rocm-utils rocm-dev
        
        echo ""
        echo "✅ ROCm installation completed!"
        echo ""
        echo "🔍 Verifying ROCm installation..."
        if command -v rocminfo &> /dev/null; then
            echo "✅ ROCm is now installed!"
            rocminfo
        else
            echo "❌ ROCm installation may have failed."
            echo "Please check the error messages above."
        fi
    else
        echo "ℹ️  Skipping ROCm installation. Service will run in CPU mode."
    fi
fi

# Check if virtual environment already exists
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

# Create virtual environment
echo ""
echo "📦 Step 3: Creating virtual environment..."
echo "----------------------------------------"

if [ ! -d ".venv" ]; then
    echo "📦 Creating new virtual environment..."
    python3 -m venv .venv
    if [ $? -eq 0 ]; then
        echo "✅ Virtual environment created successfully!"
    else
        echo "❌ Failed to create virtual environment."
        exit 1
    fi
else
    echo "✅ Virtual environment already exists."
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Verify virtual environment is activated
VENV_PYTHON=$(which python)
echo "✅ Virtual environment activated: $VENV_PYTHON"

# Verify virtual environment is working
if [[ "$VENV_PYTHON" == *".venv"* ]]; then
    echo "✅ Virtual environment is properly activated!"
else
    echo "❌ Virtual environment activation failed!"
    exit 1
fi

# Check if ROCm is available (after virtual environment is activated)
echo ""
echo "🔍 Step 4: Checking GPU support..."
echo "----------------------------------------"
echo "🔍 Checking ROCm availability..."
if python -c "import torch; print('ROCm available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✅ ROCm is available"
else
    echo "⚠️  ROCm not available. Installing CPU-only PyTorch..."
    CUDA_INDEX=""
fi

# Upgrade pip
echo ""
echo "⬆️  Step 5: Installing dependencies..."
echo "----------------------------------------"
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies."
    exit 1
fi

# Install PyTorch with ROCm support (as tested)
echo "🚀 Installing PyTorch with ROCm support..."
echo "Note: This may take a while as it downloads ROCm-enabled PyTorch packages..."

# First, uninstall any existing PyTorch packages to avoid conflicts
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
if [ $? -eq 0 ]; then
    echo "✅ PyTorch with ROCm installed successfully!"
else
    echo "❌ Failed to install PyTorch with ROCm."
    echo "Trying alternative installation method..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall
    if [ $? -eq 0 ]; then
        echo "✅ PyTorch with ROCm installed successfully (force reinstall)!"
    else
        echo "❌ Failed to install PyTorch with ROCm. Please check your system."
        exit 1
    fi
fi

# Verify installation
echo ""
echo "✅ Step 6: Verifying installation..."
echo "----------------------------------------"
echo "✅ Verifying installation..."

# Check if ROCm libraries are available
echo "🔍 Checking ROCm library availability..."
if [ -f "/opt/rocm/lib/libtorch_hip.so" ] || [ -f "/usr/lib/x86_64-linux-gnu/libtorch_hip.so" ]; then
    echo "✅ ROCm libraries found on system"
else
    echo "⚠️  ROCm libraries not found. This may cause issues with GPU acceleration."
    echo "   You may need to install ROCm drivers: https://rocmdocs.amd.com/en/latest/deploy/linux/prerequisites.html"
fi

# Verify modular architecture imports
echo "🏗️  Verifying modular architecture..."
MODULAR_VERIFICATION=$(python -c "
try:
    import config
    from utils.ssml_parser import parse_ssml
    from audio.processing import initialize_zipenhancer
    from tts.engine import initialize_environment
    from api.models import TTSRequest
    from api.routes import router
    from main import main
    print('✅ All modular components imported successfully!')
except ImportError as e:
    print(f'❌ Modular architecture import failed: {e}')
    exit(1)
" 2>&1)

echo "$MODULAR_VERIFICATION"
if [[ "$MODULAR_VERIFICATION" == *"successfully"* ]]; then
    echo "✅ Modular architecture verified!"
else
    echo "❌ Modular architecture verification failed!"
    exit 1
fi

VERIFICATION_OUTPUT=$(python -c "
import torch
import fastapi
import uvicorn
import pydub
import librosa
import psutil
import sphn
import soundfile
import moshi
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  ROCm not available. Service will run in CPU mode.')
")

if [ $? -eq 0 ]; then
    echo "$VERIFICATION_OUTPUT"
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
    echo "   source .venv/bin/activate"
    echo "   python main.py"
    echo "   OR use: ./start.sh"
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
        python main.py
    else
        echo "ℹ️  You can start the service later with:"
        echo "   source .venv/bin/activate"
        echo "   python main.py"
        echo "   OR use: ./start.sh"
    fi
else
    echo "❌ Installation verification failed!"
    echo "Please check the error messages above."
    exit 1
fi 