# 🎵 GPU-Optimized Text-to-Speech Service

A high-performance, GPU-accelerated Text-to-Speech (TTS) service built with FastAPI, PyTorch, and Kyutai TTS. Features advanced GPU memory management, batch processing, audio cleaning, and SSML support.

> **✅ Tested Hardware**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1150) with ROCm 6.3

## 🚀 Features

### Core Features
- **GPU-Accelerated TTS**: Leverages NVIDIA GPUs for fast audio generation
- **Advanced Memory Management**: Sophisticated GPU memory pooling and tensor reuse
- **Batch Processing**: Concurrent processing of multiple text segments
- **Audio Cleaning Pipeline**: GPU-accelerated noise reduction and audio enhancement
- **SSML Support**: Voice control, pauses, and speech synthesis markup
- **Multiple Output Formats**: MP3 and WAV support
- **Caching System**: Intelligent audio caching for repeated requests
- **Real-time Monitoring**: GPU/CPU usage and memory tracking

### Performance Optimizations
- **CUDA Optimizations**: Flash attention, TF32, memory-efficient operations
- **Parallel Processing**: Multiple CUDA streams for concurrent operations
- **Adaptive Memory Management**: Dynamic tensor allocation based on usage patterns
- **Fast Mode**: Aggressive optimizations for speed-critical applications
- **Request Queue Management**: Limits concurrent requests to prevent overload

## 📋 Requirements

### System Requirements
- **OS**: Linux (tested on Ubuntu 22.04)
- **GPU**: AMD GPU with ROCm support (tested on AMD Strix Halo GFX1150)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for models and cache
- **Environment**: `export HSA_OVERRIDE_GFX_VERSION=11.0.0` in ~/.bashrc

### Software Requirements
- **Python**: 3.8+
- **ROCm**: 6.3+ (compatible with PyTorch)
- **PyTorch**: 2.7.1+ with ROCm support
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **TTS Model**: Kyutai TTS or alternative TTS library

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd tts-service
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

# Install dependencies
pip install -r requirements.txt
```

### 4. Environment Setup (Required for AMD GPUs)
```bash
# Add to your ~/.bashrc file for AMD GPU compatibility
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

### 5. Verify GPU Setup
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
python run-tts-service.py
```

### 2. Test the Service
```bash
# Simple text-to-speech
curl -X POST "http://localhost:8000/api/tts" \
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

#### Example Requests

**Basic TTS:**
```bash
curl -X POST "http://localhost:8000/api/tts" \
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
curl -X POST "http://localhost:8000/api/tts" \
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
curl -X POST "http://localhost:8000/api/tts" \
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
curl -X POST "http://localhost:8000/api/tts" \
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

### Other Endpoints

#### Health Check
```bash
curl "http://localhost:8000/health"
```

#### List Available Voices
```bash
curl "http://localhost:8000/api/voices"
```

#### GPU Memory Status
```bash
curl "http://localhost:8000/api/gpu-status"
```

#### Clean Audio Only
```bash
curl -X POST "http://localhost:8000/api/clean-audio" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "path/to/audio.mp3",
    "volume_boost": 6.0,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true
  }'
```

## ⚙️ Configuration

### Environment Variables
```bash
# GPU Memory Fraction (0.0-1.0)
export CUDA_MEMORY_FRACTION=0.95

# Max Concurrent Requests
export MAX_CONCURRENT_REQUESTS=4

# Enable/Disable Features
export ENABLE_FAST_MODE=true
export ENABLE_BATCH_PROCESSING=true
export ENABLE_AUDIO_CLEANING=true
```

### Performance Settings

#### Fast Mode Configuration
- **Batch Size**: 24 (vs 16 in normal mode)
- **CUDA Streams**: 16 (vs 8 in normal mode)
- **GPU Memory Fraction**: 95% (vs 90% in normal mode)
- **Audio Cleaning**: Enabled with optimized settings

#### Memory Management
- **GPU Memory Pool**: 81.6GB allocation
- **Tensor Reuse**: Automatic cleanup and reuse
- **Adaptive Allocation**: Based on usage patterns

## 📊 Performance Metrics

### Typical Performance (AMD Strix Halo GFX1150)
- **Simple Text (50 words)**: ~5-8 seconds
- **Complex SSML (200 words)**: ~15-25 seconds
- **Fast Mode**: 20-30% faster than normal mode
- **GPU Utilization**: 4-5% during processing
- **Memory Usage**: ~4GB GPU, ~54% system RAM

### Optimization Results
- **88% Speed Improvement**: From 131s to ~19s baseline
- **GPU Memory Efficiency**: 95% utilization with pooling
- **Concurrent Processing**: Up to 4 simultaneous requests
- **Audio Quality**: Maintained with cleaning pipeline

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
# Reduce GPU memory fraction
export CUDA_MEMORY_FRACTION=0.80
```

**2. Slow Performance**
```bash
# Enable fast mode
curl -X POST "http://localhost:8000/api/tts" \
  -d '{"text": "test", "fast_mode": true}'
```

**3. Audio Quality Issues**
```bash
# Enable audio cleaning
curl -X POST "http://localhost:8000/api/tts" \
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
tts-service/
├── run-tts-service.py    # Main service file
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

- **Kyutai TTS**: Core TTS model
- **PyTorch**: GPU acceleration framework
- **FastAPI**: Web framework
- **CUDA**: NVIDIA GPU computing platform

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Include system specifications and error details

---

**Last Updated**: July 21, 2025  
**Version**: 1.0.0  
**GPU Optimized**: ✅  
**Production Ready**: ✅ 