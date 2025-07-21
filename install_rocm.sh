#!/bin/bash

# ROCm Installation Helper Script
# This script helps install ROCm drivers for AMD GPUs

echo "🔧 ROCm Installation Helper"
echo "==========================="
echo ""
echo "This script will help you install ROCm drivers for AMD GPUs."
echo "ROCm is required for GPU acceleration with AMD GPUs."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs to be run as root (sudo) to install ROCm."
    echo "Please run: sudo ./install_rocm.sh"
    exit 1
fi

echo "🔍 Checking system requirements..."

# Check Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "✅ Ubuntu version: $UBUNTU_VERSION"

# Check if ROCm is already installed
if command -v rocminfo &> /dev/null; then
    echo "✅ ROCm is already installed!"
    rocminfo
    exit 0
fi

echo ""
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
echo "🔍 Verifying installation..."
if command -v rocminfo &> /dev/null; then
    echo "✅ ROCm is now installed!"
    rocminfo
else
    echo "❌ ROCm installation may have failed."
    echo "Please check the error messages above."
fi

echo ""
echo "📚 For more information, see:"
echo "   https://rocmdocs.amd.com/en/latest/deploy/linux/prerequisites.html"
echo ""
echo "🚀 After installing ROCm, you can run the TTS service installation:"
echo "   ./install.sh" 