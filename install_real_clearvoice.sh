#!/bin/bash

# Real ClearerVoice-Studio Installation Script
# This script installs the actual ClearerVoice-Studio from the official repository

set -e  # Exit on any error

echo "🎵 Real ClearerVoice-Studio Installation Script"
echo "==============================================="
echo ""
echo "This script will install the actual ClearerVoice-Studio from:"
echo "https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install git if not available
if ! command -v git &> /dev/null; then
    echo "📦 Installing git..."
    sudo apt-get update
    sudo apt-get install -y git
fi

# Clone ClearerVoice-Studio repository
echo "📥 Cloning ClearerVoice-Studio repository..."
if [ ! -d "clearvoice_studio" ]; then
    git clone https://github.com/modelscope/ClearerVoice-Studio.git clearvoice_studio
    echo "✅ ClearerVoice-Studio repository cloned"
else
    echo "✅ ClearerVoice-Studio repository already exists"
fi

# Install ClearerVoice-Studio dependencies
echo "📦 Installing ClearerVoice-Studio dependencies..."
cd clearvoice_studio

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install additional dependencies
pip install modelscope transformers accelerate diffusers

echo ""
echo "✅ ClearerVoice-Studio installation completed!"
echo ""

# Test ClearerVoice-Studio availability
echo "🧪 Testing ClearerVoice-Studio availability..."
cd ..
python3 -c "
import sys
sys.path.insert(0, './clearvoice_studio/clearvoice')
try:
    from clearvoice import ClearerVoice
    print('✅ Real ClearerVoice-Studio is available')
except ImportError as e:
    print(f'❌ ClearerVoice-Studio not available: {e}')
    print('🔄 Will use modelscope fallback')
"

echo ""
echo "🎉 Real ClearerVoice-Studio installation completed!"
echo ""
echo "📚 Next steps:"
echo "1. Start the service: ./start.sh"
echo "2. Test ClearerVoice enhancement:"
echo "   curl -X POST 'http://localhost:7861/api/tts' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"text\": \"test\", \"clearvoice_enhancement\": true}'"
echo ""
echo "📖 Everything is now integrated in kyutai-tts-service.py" 