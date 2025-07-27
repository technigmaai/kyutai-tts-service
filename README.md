# 🎵 GPU-Optimized Text-to-Speech Service

A high-performance, GPU-accelerated Text-to-Speech (TTS) service built with FastAPI, PyTorch, and Moshi TTS. Features advanced GPU memory management, batch processing, audio cleaning, and SSML support. **Now with real ClearerVoice-Studio integration for professional-grade audio enhancement.**

> **✅ Tested Hardware**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1150) with ROCm 6.3

## 📚 Documentation

- **[Audio Cleaning Parameters Guide](AUDIO_CLEANING_PARAMETERS.md)** - Detailed explanation of how audio cleaning parameters work
- **[ClearerVoice-Studio Integration](CLEARVOICE_INTEGRATION.md)** - Real ClearerVoice-Studio integration guide

## 🚀 Features

### Core Features
- **GPU-Accelerated TTS**: Leverages AMD GPUs with ROCm for fast audio generation
- **Real ClearerVoice-Studio Integration**: Professional-grade audio enhancement using the official repository
- **ZipEnhancer Noise Suppression**: Advanced acoustic noise suppression using ModelScope's ZipEnhancer
- **Advanced Memory Management**: Sophisticated GPU memory pooling and tensor reuse
- **Batch Processing**: Concurrent processing of multiple text segments
- **Audio Cleaning Pipeline**: GPU-accelerated noise reduction and audio enhancement
- **SSML Support**: Voice control, pauses, and speech synthesis markup
- **Multiple Output Formats**: MP3 and WAV support
- **Caching System**: Intelligent audio caching for repeated requests
- **Real-time Monitoring**: GPU/CPU usage and memory tracking

### Performance Optimizations
- **ROCm Optimizations**: Flash attention, TF32, memory-efficient operations
- **Parallel Processing**: Multiple GPU streams for concurrent operations
- **Adaptive Memory Management**: Dynamic tensor allocation based on usage patterns
- **Fast Mode**: Aggressive optimizations for speed-critical applications
- **Request Queue Management**: Limits concurrent requests to prevent overload

## 📋 Requirements

### System Requirements
- **OS**: Ubuntu 24.04 (tested)
- **Python**: 3.12.10 (tested)
- **GPU**: AMD GPU with ROCm support (tested on AMD Strix Halo GFX1150)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and cache
- **Environment**: `export HSA_OVERRIDE_GFX_VERSION=11.0.0` in ~/.bashrc

### Software Requirements
- **Python**: 3.12.10 (tested)
- **ROCm**: 6.3+ (AMD GPU computing platform)
- **PyTorch**: 2.7.1+ with ROCm support (AMD GPU optimized)
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **TTS Model**: Moshi TTS (kyutai/tts-1.6b-en_fr)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/technigmaai/kyutai-tts-service.git
cd kyutai-tts-service
```

### 2. Quick Installation (Recommended)
```bash
# Run the automated installation script
./install.sh
```

### 3. Manual Installation (Alternative)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install other dependencies (PyTorch packages are excluded from requirements.txt)
pip install -r requirements.txt

# Install PyTorch with ROCm support (as tested)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

### 4. Environment Setup (Required for AMD GPUs)
```bash
# Add to your ~/.bashrc file for AMD GPU compatibility
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

### 5. ROCm Installation (Required for GPU Acceleration)
The installer will automatically check for ROCm and offer to install it if needed. If you encounter ROCm library errors:

```bash
# Option 1: Use the comprehensive installer (recommended)
sudo ./install.sh

# Option 2: Manual installation
# Follow the official guide: https://rocmdocs.amd.com/en/latest/deploy/linux/prerequisites.html
```

### 6. Verify GPU Setup
```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 🚀 Quick Start

### 1. Start the Service
```bash
# Quick start (recommended)
./start.sh

# Or manual start
source .venv/bin/activate
python kyutai-tts-service.py
```

### 2. Test the Service
```bash
# Simple text-to-speech
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the TTS service!",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "apply_cleaning": true
  }' \
  --output test_output.mp3
```

## 📚 API Documentation

> **📖 Audio Cleaning Guide**: For detailed information about audio cleaning parameters, see the **[Audio Cleaning Parameters Guide](AUDIO_CLEANING_PARAMETERS.md)**.

### Endpoint: `POST /api/tts`

Generate speech from text with advanced options.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **required** | Text to convert to speech |
| `voice_choice` | string | "Happy" | Voice to use (Happy, Sad, Angry, etc.) |
| `output_format` | string | "mp3" | Output format (mp3, wav) |
| `apply_cleaning` | boolean | false | Enable audio cleaning pipeline |
| `volume_boost` | float | 6.0 | Audio volume boost (0-20) |
| `remove_crackles` | boolean | true | Remove audio crackles |
| `apply_filters` | boolean | true | Apply audio filters |
| `reduce_noise` | boolean | true | Reduce background noise |
| `fast_mode` | boolean | false | Enable fast mode (aggressive optimizations) |
| `filename` | string | null | Custom output filename |
| `use_native_sample_rate` | boolean | true | Use model's native sample rate instead of 44.1kHz |
| `use_clearvoice` | boolean | true | Use ClearerVoice-Studio for audio enhancement (if available) |
| `clearvoice_enhancement` | boolean | false | Enable ClearerVoice enhancement (overrides apply_cleaning) |
| `apply_zipenhancer` | boolean | false | Enable ZipEnhancer noise suppression post-processing |

#### Example Requests

**Basic TTS:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world!",
    "voice_choice": "Happy",
    "output_format": "mp3"
  }' \
  --output hello.mp3
```

**Advanced TTS with Cleaning:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is high-quality audio with noise reduction.",
    "voice_choice": "Happy",
    "output_format": "wav",
    "apply_cleaning": true,
    "volume_boost": 8.0,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true,
    "filename": "clean_audio.wav"
  }' \
  --output clean_audio.wav
```

**Fast Mode (Speed Optimized):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Quick generation with optimized settings.",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "fast_mode": true,
    "filename": "fast_output.mp3"
  }' \
  --output fast_output.mp3
```

**SSML with Multiple Voices:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<speak><voice name=\"Happy\">Hello! This is a happy voice.</voice><break time=\"1s\"/><voice name=\"Sad\">And this is a sad voice.</voice><break time=\"2s\"/><voice name=\"Angry\">Finally, an angry voice!</voice></speak>",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "apply_cleaning": true,
    "filename": "multi_voice.mp3"
  }' \
  --output multi_voice.mp3
```

**Native Sample Rate (Maximum Quality):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses the model's native sample rate for maximum quality.",
    "voice_choice": "Happy",
    "output_format": "wav",
    "use_native_sample_rate": true,
    "apply_cleaning": false,
    "filename": "native_quality.wav"
  }' \
  --output native_quality.wav
```

**Standard Sample Rate (Compatibility):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses standard 44.1kHz for maximum compatibility.",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "use_native_sample_rate": false,
    "apply_cleaning": true,
    "filename": "standard_quality.mp3"
  }' \
  --output standard_quality.mp3
```

**ClearerVoice-Studio Enhancement (Professional Quality):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses ClearerVoice-Studio for professional-grade enhancement.",
    "voice_choice": "Happy",
    "output_format": "wav",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "filename": "professional_enhanced.wav"
  }' \
  --output professional_enhanced.wav
```

**Hybrid Enhancement (ClearerVoice + GPU Cleaning):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses both ClearerVoice and GPU cleaning for maximum quality.",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "apply_cleaning": true,
    "filename": "hybrid_enhanced.mp3"
  }' \
  --output hybrid_enhanced.mp3
```

**ZipEnhancer Noise Suppression (New Feature):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses ZipEnhancer for advanced acoustic noise suppression.",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "apply_zipenhancer": true,
    "filename": "zipenhancer_enhanced.mp3"
  }' \
  --output zipenhancer_enhanced.mp3
```

### Other Endpoints

#### Health Check
```bash
curl "http://localhost:7861/api/health"
```

#### List Available Voices
```bash
curl "http://localhost:7861/api/voices"
```

#### Memory Status
```bash
curl "http://localhost:7861/api/memory/stats"
```

#### Clean Audio Only
```bash
curl -X POST "http://localhost:7861/api/clean-audio" \
  -F "audio_file=@path/to/audio.mp3" \
  -F "volume_boost=6.0" \
  -F "remove_crackles=true" \
  -F "apply_filters=true" \
  -F "reduce_noise=true"
```

#### ClearerVoice-Studio Enhancement
```bash
curl -X POST "http://localhost:7861/api/clearvoice-enhance" \
  -F "audio_file=@path/to/audio.mp3" \
  -F "use_clearvoice=true"
```

#### ZipEnhancer Status Check
```bash
curl "http://localhost:7861/api/zipenhancer/status"
```

## ⚙️ Configuration

### Performance Settings

The service uses the following default configuration values:

```python
# Request Management
MAX_CONCURRENT_REQUESTS = 4
MAX_BATCH_SIZE = 16

# GPU Optimizations
ENABLE_FAST_MODE = True
ENABLE_BATCH_PROCESSING = True
ENABLE_CONCURRENT_CLEANING = True
ENABLE_MODEL_QUANTIZATION = True
ENABLE_PARALLEL_PROCESSING = True

# Memory Management
GPU_MEMORY_FRACTION = 0.90  # 90% of GPU memory
MEMORY_POOL_MAX_FRACTION = 0.85  # 85% for memory pool
```

### Performance Settings

#### Fast Mode Configuration
- **Batch Size**: 24 (vs 16 in normal mode)
- **GPU Memory Fraction**: 90% (configurable)
- **Audio Cleaning**: Enabled with optimized settings

#### Memory Management
- **GPU Memory Pool**: Adaptive allocation based on available GPU memory
- **Tensor Reuse**: Automatic cleanup and reuse
- **Adaptive Allocation**: Based on usage patterns

## 🔧 Advanced Usage

### SSML Support
The service supports Speech Synthesis Markup Language (SSML) for advanced voice control:

```xml
<speak>
  <voice name="Happy">Hello! This is a happy voice.</voice>
  <break time="1s"/>
  <voice name="Sad">This is a sad voice.</voice>
  <break time="2s"/>
  <voice name="Angry">And this is an angry voice!</voice>
</speak>
```

### Batch Processing
For multiple text segments, the service automatically:
1. Parses SSML into segments
2. Processes segments in batches
3. Reconstructs audio in original order
4. Applies final cleaning and normalization

### Audio Cleaning Pipeline
The GPU-accelerated cleaning includes:
- **Noise Reduction**: Background noise removal
- **Crackle Removal**: Audio artifact cleanup
- **Volume Normalization**: Consistent audio levels
- **Filter Application**: High-quality audio filters

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# The service automatically manages GPU memory
# Check memory usage via API
curl "http://localhost:7861/api/memory/stats"
```

**2. Slow Performance**
```bash
# Enable fast mode
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "fast_mode": true}'
```

**3. Audio Quality Issues**
```bash
# Enable audio cleaning
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "apply_cleaning": true}'
```

**4. Service Not Starting**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check dependencies
pip list | grep torch

# Verify HSA environment variable
echo $HSA_OVERRIDE_GFX_VERSION
```

### Log Analysis
The service provides detailed logging:
- **GPU Usage**: Real-time memory and utilization
- **Processing Time**: Per-segment and total timing
- **File Operations**: Temporary file creation and cleanup
- **Error Tracking**: Detailed error messages and stack traces

## 📁 File Structure

```
kyutai-tts-service/
├── kyutai-tts-service.py    # Main service file
├── README.md            # This file
├── .gitignore          # Git ignore rules
├── requirements.txt     # Python dependencies
├── install.sh          # Automated installation script
└── start.sh            # Quick start script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Moshi TTS**: Core TTS model (kyutai/tts-1.6b-en_fr)
- **PyTorch**: GPU acceleration framework
- **FastAPI**: Web framework
- **ROCm**: AMD GPU computing platform

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Include system specifications and error details

---

**Last Updated**: July 21, 2025  
**Version**: 3.0.0  
**GPU Optimized**: ✅  
**Production Ready**: ✅ 