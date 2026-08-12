# 🛠️ Installation

This guide covers installing the Kyutai TTS Service from scratch on an AMD GPU
system with ROCm. It uses **uv** (fast Python package manager) and **AMD-built
PyTorch wheels** for native gfx1151 (Strix Halo) support.

---

## Requirements

### System Requirements
- **OS**: Ubuntu 24.04+ (tested on Ubuntu 26.04 / EVO-X2)
- **Python**: 3.13 (required — AMD ROCm wheels are cp310–cp313 only)
- **GPU**: AMD GPU with ROCm support (tested on AMD Strix Halo GFX1151 / Radeon 8060S)
- **ROCm**: 7.x (tested with 7.1.4)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and cache
- **ffmpeg**: required by pydub for MP3 export (static build in `~/bin` works)

> **No `HSA_OVERRIDE_GFX_VERSION` hack needed anymore.** Modern ROCm (7.x) and
> the AMD-built PyTorch wheels support gfx1151 (Strix Halo) natively.

### Software Requirements
- **Python**: 3.13
- **ROCm**: 7.x (AMD GPU computing platform)
- **PyTorch**: 2.9.1+rocm7.2.4 (AMD-built wheel from repo.radeon.com)
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **TTS Model**: Moshi TTS (kyutai/tts-1.6b-en_fr)

> **Why AMD-built PyTorch wheels?** The official PyTorch ROCm wheels do **not**
> include gfx1151 (Strix Halo) kernels. AMD's own wheels at `repo.radeon.com`
> do. torch is pinned to 2.9.x because moshi (the TTS library) requires
> `torch<2.10`.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/technigmaai/kyutai-tts-service.git
cd kyutai-tts-service
```

### 2. Quick Installation (Recommended)
```bash
# Install uv (fast Python package manager, no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.13
uv venv --python 3.13 .venv

# Install AMD-built PyTorch stack (gfx1151 support)
VIRTUAL_ENV=.venv uv pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchaudio-2.9.0%2Brocm7.2.4.gite3c6ee2b-cp313-cp313-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchvision-0.24.0%2Brocm7.2.4.gitb919bd0c-cp313-cp313-linux_x86_64.whl

# Install the rest of the dependencies
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# Install ffmpeg (static build, no sudo needed) for pydub MP3 export
mkdir -p ~/bin
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ
cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffprobe ~/bin/
```

> Alternatively, run `./install.sh` which automates all of the above (checks
> ROCm/GPU, creates the venv, installs the AMD wheels, ffmpeg, and verifies).

### 3. Environment Setup (Required for AMD GPUs)
```bash
# ROCm library path for AMD-built PyTorch wheels (libroctx64.so.4 etc.)
# Must be set BEFORE launching Python — start.sh does this automatically.
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"
```

### 4. ROCm Installation (Required for GPU Acceleration)
ROCm must be installed system-wide for GPU acceleration. The installer
(`./install.sh`) checks for ROCm and reports if it's missing.

```bash
# Manual installation
# Follow the official guide: https://rocm.docs.amd.com/en/latest/deploy/linux/install.html
```

### 5. Verify GPU Setup
```bash
# From the repo directory, with the venv active
LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib" .venv/bin/python -c "
import torch
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Arch: {torch.cuda.get_device_properties(0).gcnArchName}')
"
```

Expected output on EVO-X2:
```
ROCm available: True
GPU: AMD Radeon 8060S Graphics
Arch: gfx1151
```

---

## Related

- [Back to README](../README.md)
- [Quick Start](../README.md#-quick-start)
- [Troubleshooting](TROUBLESHOOTING.md)
