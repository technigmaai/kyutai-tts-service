# Requirements Analysis for TTS Service

## 🖥️ Tested Hardware
- **GPU**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1150)
- **Platform**: ROCm 6.3
- **Environment**: PyTorch2 (.venv)
- **HSA Override**: `export HSA_OVERRIDE_GFX_VERSION=11.0.0` in ~/.bashrc
- **PyTorch Installation**: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3`

## 🔍 Analysis of PyTorch2 Environment

Based on the analysis of `~/Development/PyTorch2/.venv/bin/activate`, here are the findings:

### ✅ Available Packages (Matching Requirements)

| Package | Required Version | Available Version | Status |
|---------|------------------|-------------------|---------|
| torch | >=2.0.0 | 2.7.1+rocm6.3 | ✅ Available |
| torchvision | >=0.15.0 | 0.22.1+rocm6.3 | ✅ Available |
| torchaudio | >=2.0.0 | 2.7.1+rocm6.3 | ✅ Available |
| fastapi | >=0.100.0 | 0.116.1 | ✅ Available |
| uvicorn | >=0.20.0 | 0.35.0 | ✅ Available |
| pydantic | >=2.0.0 | 2.11.7 | ✅ Available |
| psutil | >=5.9.0 | 7.0.0 | ✅ Available |
| pydub | >=0.25.0 | 0.25.1 | ✅ Available |
| librosa | >=0.10.0 | 0.11.0 | ✅ Available |
| numpy | >=1.24.0 | 1.26.4 | ✅ Available |
| scipy | >=1.10.0 | 1.16.0 | ✅ Available |
| requests | >=2.31.0 | 2.32.4 | ✅ Available |
| python-multipart | >=0.0.6 | 0.0.20 | ✅ Available |

### ❌ Missing Packages

| Package | Required Version | Status |
|---------|------------------|---------|
| kyutai-tts | >=0.1.0 | ❌ **NOT AVAILABLE** |

### 🔧 Key Changes Made

1. **Updated PyTorch Versions**: Changed from CUDA to ROCm support
   - `torch>=2.0.0` → `torch>=2.7.1+rocm6.3`
   - `torchvision>=0.15.0` → `torchvision>=0.22.1+rocm6.3`
   - `torchaudio>=2.0.0` → `torchaudio>=2.7.1+rocm6.3`

2. **Updated All Dependencies**: Bumped versions to match PyTorch2 environment
   - `fastapi>=0.100.0` → `fastapi>=0.116.1`
   - `uvicorn[standard]>=0.20.0` → `uvicorn[standard]>=0.35.0`
   - `pydantic>=2.0.0` → `pydantic>=2.11.7`
   - `psutil>=5.9.0` → `psutil>=7.0.0`
   - `pydub>=0.25.0` → `pydub>=0.25.1`
   - `librosa>=0.10.0` → `librosa>=0.11.0`
   - `numpy>=1.24.0` → `numpy>=1.26.4`
   - `scipy>=1.10.0` → `scipy>=1.16.0`
   - `requests>=2.31.0` → `requests>=2.32.4`
   - `python-multipart>=0.0.6` → `python-multipart>=0.0.20`

3. **Commented Out kyutai-tts**: Since it's not available in the PyTorch2 environment
   - Added note about needing to install separately
   - Suggested alternative TTS libraries

4. **Updated Installation Script**: Modified `install.sh` to use ROCm instead of CUDA
   - Changed PyTorch installation to use ROCm index
   - Updated version specifications

5. **Updated Documentation**: Modified README.md to reflect ROCm setup
   - Changed CUDA references to ROCm
   - Updated system requirements
   - Updated verification commands

## 🚨 Important Notes

### TTS Model Issue
The **kyutai-tts** package is **NOT AVAILABLE** in the PyTorch2 environment. This means:

1. **Current Service Won't Work**: The TTS service depends on kyutai-tts
2. **Need Alternative**: Either install kyutai-tts separately or use an alternative TTS library
3. **Options**:
   - `pip install kyutai-tts` (if available from PyPI)
   - Use alternative TTS libraries like:
     - `gTTS` (Google Text-to-Speech)
     - `pyttsx3` (offline TTS)
     - `TTS` (Coqui TTS)
     - `transformers` + TTS models

### ROCm vs CUDA
The PyTorch2 environment uses **ROCm** instead of **CUDA**:
- **ROCm**: AMD's GPU computing platform (tested on AMD Strix Halo GFX1150)
- **CUDA**: NVIDIA's GPU computing platform
- **Compatibility**: ROCm is compatible with AMD GPUs, CUDA with NVIDIA GPUs

## 📋 Recommendations

### Immediate Actions
1. **Test TTS Model**: Try installing kyutai-tts separately
2. **Alternative TTS**: Research and implement alternative TTS libraries
3. **GPU Compatibility**: Verify ROCm compatibility with your GPU

### Long-term Considerations
1. **Environment Choice**: Decide between PyTorch2 (ROCm) or CUDA environment
2. **TTS Library**: Choose the most suitable TTS library for your needs
3. **Documentation**: Update all references to reflect the chosen setup

## 🔄 Next Steps

1. **Install kyutai-tts**: `pip install kyutai-tts`
2. **Test Installation**: Verify the TTS service works with the PyTorch2 environment
3. **Update Code**: Modify the TTS service code if needed for ROCm compatibility
4. **Documentation**: Update README with final working setup

## ⚙️ AMD GPU Setup Requirements

### HSA Environment Variable
For AMD GPU compatibility, the following environment variable must be set:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

**Purpose**: This override tells the AMD drivers to use GFX11.0.0 architecture, which is compatible with the AMD Strix Halo GPU (GFX1150).

**Setup**: Add to your `~/.bashrc` file:
```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

**Verification**: Check if the variable is set:
```bash
echo $HSA_OVERRIDE_GFX_VERSION
```

### PyTorch Installation Method
The tested PyTorch installation command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

**Note**: This command installs the latest compatible versions for ROCm 6.3, which were verified to work with the AMD Strix Halo GPU.

---

**Analysis Date**: July 21, 2025  
**Environment**: PyTorch2 (.venv)  
**GPU Platform**: ROCm 6.3  
**Tested Hardware**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1150)  
**Status**: ⚠️ Requires TTS model installation 