# 🎵 GPU-Optimized Text-to-Speech Service (Modular Architecture)

A high-performance, GPU-accelerated Text-to-Speech (TTS) service built with FastAPI, PyTorch, and Moshi TTS. Features ZipEnhancer noise suppression, SSML support for multiple voices, and optimized processing for AMD GPUs with ROCm. **Now with modular architecture for better maintainability and scalability.**

> **✅ Tested Hardware**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1150) with ROCm 6.3

## 🏗️ Architecture

This service uses a **modular architecture** with clear separation of concerns:

- **`main.py`**: Entry point and service orchestration
- **`config.py`**: Configuration constants and defaults
- **`api/`**: FastAPI routes and request/response models
- **`tts/`**: TTS model loading and audio generation
- **`audio/`**: ZipEnhancer noise suppression and audio effects
- **`utils/`**: SSML parsing and utility functions

## 🚀 Features

### Core Features
- **GPU-Accelerated TTS**: Leverages AMD GPUs with ROCm for fast audio generation
- **ZipEnhancer Noise Suppression**: Advanced acoustic noise suppression with windowed processing for superior quality
- **Audio Processing Effects**: Normalization, volume boost, fade-in/out effects, and custom bitrate control
- **SSML Support**: Voice control, pauses, and speech synthesis markup for multiple voices
- **20+ Voice Options**: Various emotional and stylistic voices (Happy, Sad, Angry, Calm, etc.)
- **Flexible Output**: MP3/WAV formats with custom filenames and quality settings
- **Memory Management**: Automatic temporary file cleanup and GPU optimization
- **Modular Design**: Clean, maintainable code structure with separated components

### Performance Optimizations
- **ROCm Integration**: Optimized for AMD GPUs with proper environment configuration
- **Torch Optimization**: Multi-threaded processing with 8 CPU threads
- **Windowed Processing**: Memory-efficient processing for large audio files
- **Quality Modes**: Three processing levels (standard/high/ultra) for different use cases
- **Component Isolation**: Independent modules for easier testing and maintenance

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
# Run the automated installation script (now with modular architecture validation)
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
# Quick start (recommended) - uses modular main.py
./start.sh

# Or manual start with modular architecture
source .venv/bin/activate
python main.py
```

### 2. Test the Service
```bash
# Simple text-to-speech
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the modular TTS service!",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "apply_zipenhancer": true
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
| `output_format` | string | "mp3" | Output format: "mp3" or "wav" |
| `filename` | string | null | Custom output filename (extension auto-added) |
| `apply_zipenhancer` | boolean | false | Enable ZipEnhancer noise suppression post-processing |
| `zipenhancer_quality` | string | "high" | ZipEnhancer quality mode: "standard", "high", "ultra" |
| `zipenhancer_window_size` | float | 2.0 | Window size in seconds for windowed processing (1.0-5.0) |
| `normalize` | boolean | false | Apply audio normalization |
| `volume_boost` | float | null | Volume boost in decibels (e.g., 5.0, -3.0) |
| `fade_in` | integer | null | Fade-in duration in milliseconds |
| `fade_out` | integer | null | Fade-out duration in milliseconds |
| `bitrate` | string | "320k" | MP3 bitrate (e.g., "128k", "192k", "320k") |

#### Example Requests

**Basic TTS (MP3):**
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

**Basic TTS (WAV):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world in high quality WAV format!",
    "voice_choice": "Happy",
    "output_format": "wav"
  }' \
  --output hello.wav
```

**Custom Filename:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will have a custom filename!",
    "voice_choice": "Happy",
    "output_format": "mp3",
    "filename": "my_custom_audio"
  }' \
  --output my_custom_audio.mp3
```

**SSML with Multiple Voices:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<speak><voice name=\"Happy\">Hello! This is a happy voice.</voice><break time=\"1s\"/><voice name=\"Sad\">And this is a sad voice.</voice><break time=\"2s\"/><voice name=\"Angry\">Finally, an angry voice!</voice></speak>",
    "voice_choice": "Happy"
  }' \
  --output multi_voice.mp3
```

**ZipEnhancer Noise Suppression (Advanced Quality - WAV):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses ZipEnhancer with high-quality windowed processing.",
    "voice_choice": "Happy",
    "output_format": "wav",
    "apply_zipenhancer": true,
    "zipenhancer_quality": "high",
    "zipenhancer_window_size": 2.0
  }' \
  --output zipenhancer_high_quality.wav
```

**ZipEnhancer Fast Mode (Standard Quality):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This uses ZipEnhancer in fast mode for quick processing.",
    "voice_choice": "Happy",
    "apply_zipenhancer": true,
    "zipenhancer_quality": "standard",
    "filename": "enhanced_fast_audio"
  }' \
  --output enhanced_fast_audio.mp3
```

**ZipEnhancer Ultra Quality (Small Windows):**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This uses ZipEnhancer with ultra quality and smaller processing windows for maximum detail.",
    "voice_choice": "Happy",
    "apply_zipenhancer": true,
    "zipenhancer_quality": "ultra",
    "zipenhancer_window_size": 1.0
  }' \
  --output zipenhancer_ultra.mp3
```

**Audio Processing Effects:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio has normalization, volume boost, and fade effects applied.",
    "voice_choice": "Happy",
    "normalize": true,
    "volume_boost": 5.0,
    "fade_in": 2000,
    "fade_out": 3000,
    "bitrate": "192k"
  }' \
  --output processed_audio.mp3
```

**Complete Enhancement Pipeline:**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Professional audio with ZipEnhancer and post-processing effects.",
    "voice_choice": "Happy",
    "output_format": "wav",
    "apply_zipenhancer": true,
    "zipenhancer_quality": "high",
    "normalize": true,
    "volume_boost": 3.0,
    "fade_in": 1500,
    "fade_out": 2000,
    "filename": "professional_enhanced_audio"
  }' \
  --output professional_enhanced_audio.wav
```

### Other Endpoints

#### ZipEnhancer Status Check
```bash
curl "http://localhost:7861/api/zipenhancer/status"
# Returns: Available quality modes, configuration options, and status
```

## ⚙️ Configuration

### Performance Settings

The service is pre-configured for optimal performance with:
- **GPU Acceleration**: Automatic ROCm utilization for AMD GPUs
- **Memory Management**: Automatic temporary file cleanup
- **Torch Optimization**: 8 threads for CPU operations, optimized interop threads

### Modular Configuration

Configuration is centralized in `config.py`:
- **Model Settings**: TTS model repository and voice options
- **GPU Settings**: ROCm environment variables and optimization
- **Audio Settings**: Default formats, bitrates, and sample rates
- **ZipEnhancer Settings**: Quality modes and processing parameters
- **Server Settings**: Host and port configuration

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

### SSML Processing
For multiple text segments, the service automatically:
1. Parses SSML into voice segments and pauses (handled by `utils/ssml_parser.py`)
2. Processes each segment with the specified voice (handled by `tts/engine.py`)
3. Reconstructs audio in original order
4. Applies ZipEnhancer if requested (handled by `audio/processing.py`)

### ZipEnhancer Quality Modes
The service offers three ZipEnhancer processing modes:

#### **Standard Mode** (`zipenhancer_quality: "standard"`)
- **Processing**: Simple, single-pass enhancement
- **Speed**: Fastest processing time
- **Memory**: Low memory usage
- **Best for**: Quick processing, shorter audio files

#### **High Mode** (`zipenhancer_quality: "high"`) - **Default**
- **Processing**: Windowed processing with 2-second chunks
- **Speed**: Moderate processing time
- **Memory**: Efficient memory usage for large files
- **Quality**: Better noise suppression and artifact removal
- **Best for**: Most use cases, balanced quality/speed

#### **Ultra Mode** (`zipenhancer_quality: "ultra"`)
- **Processing**: Same as High with optimized settings
- **Speed**: Similar to High mode
- **Quality**: Maximum noise suppression quality
- **Best for**: Professional audio, critical applications

### Window Size Configuration
- **Default**: 2.0 seconds per processing window
- **Range**: 1.0 - 5.0 seconds recommended
- **Smaller windows**: Better quality, more processing time
- **Larger windows**: Faster processing, potential quality trade-offs

## 🏗️ Modular Architecture Details

The service is organized into logical modules for better maintainability:

### **Entry Point** (`main.py`)
- Service initialization and startup
- Model loading coordination
- FastAPI app configuration
- Error handling and logging setup

### **Configuration** (`config.py`)
- All configuration constants and defaults
- Voice options and model repositories
- GPU environment settings
- Performance optimizations

### **API Layer** (`api/`)
- **`models.py`**: Pydantic request/response models
- **`routes.py`**: FastAPI endpoint handlers and routing

### **TTS Engine** (`tts/engine.py`)
- TTS model loading and initialization
- Audio generation from text/SSML
- Voice processing and management
- GPU optimization settings

### **Audio Processing** (`audio/processing.py`)
- ZipEnhancer noise suppression
- Audio effects (normalization, volume, fades)
- Windowed processing for large files
- Audio format conversion

### **Utilities** (`utils/ssml_parser.py`)
- SSML parsing and validation
- WAV header creation for raw audio
- Text sanitization and processing

## 🐛 Troubleshooting

### Common Issues

**1. GPU Memory Issues**
```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**2. Slow Performance**
```bash
# Use standard mode for faster processing
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "apply_zipenhancer": true, "zipenhancer_quality": "standard"}'
```

**3. Audio Quality Issues**
```bash
# Enable ZipEnhancer for better quality
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "apply_zipenhancer": true, "zipenhancer_quality": "high"}'
```

**4. Service Not Starting**
```bash
# Check if all modular components are present
python -c "from main import main; print('✅ Modular architecture working')"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check dependencies
pip list | grep torch

# Verify HSA environment variable
echo $HSA_OVERRIDE_GFX_VERSION
```

**5. Import Errors (Modular Architecture)**
```bash
# Verify all modules can be imported
python -c "
import config
from utils.ssml_parser import parse_ssml
from audio.processing import initialize_zipenhancer
from tts.engine import initialize_environment
from api.models import TTSRequest
from api.routes import router
print('✅ All modules imported successfully')
"
```

### Log Analysis
The service provides detailed logging:
- **GPU Usage**: Real-time memory and utilization
- **Processing Time**: Per-segment and total timing
- **File Operations**: Temporary file creation and cleanup
- **Error Tracking**: Detailed error messages and stack traces
- **Module Loading**: Component initialization status

## 📁 File Structure

```
kyutai-tts-service/
├── main.py                 # 🚀 Service entry point (modular)
├── config.py              # ⚙️ Configuration constants
├── api/                    # 🌐 API Layer
│   ├── __init__.py
│   ├── models.py          # 📝 Pydantic request/response models
│   └── routes.py          # 🛣️ FastAPI routes and handlers
├── tts/                    # 🎤 TTS Engine
│   ├── __init__.py
│   └── engine.py          # 🔧 Model loading and audio generation
├── audio/                  # 🎵 Audio Processing
│   ├── __init__.py
│   └── processing.py      # 🎛️ ZipEnhancer and audio effects
├── utils/                  # 🛠️ Utilities
│   ├── __init__.py
│   └── ssml_parser.py     # 📄 SSML parsing and WAV utilities
├── archive/                # 📦 Legacy files (safely stored)
│   ├── kyutai-tts-service.py    # Original monolithic service
│   ├── kyutai-tts-service.py.backup # Backup copy
│   └── simple.py               # Original ZipEnhancer test
├── README.md               # 📖 This documentation
├── .gitignore             # 🚫 Git ignore rules
├── requirements.txt        # 📋 Python dependencies
├── install.sh             # 🔧 Automated installation (modular-aware)
├── start.sh               # ▶️ Quick start script (modular-aware)
└── .venv/                 # 🐍 Python virtual environment
```

## 🔄 Migration from Monolithic Version

If you're upgrading from the previous monolithic version:

1. **Backup**: Original files are automatically moved to `archive/`
2. **Configuration**: Settings are now centralized in `config.py`
3. **Startup**: Use `python main.py` or `./start.sh` instead of `kyutai-tts-service.py`
4. **Functionality**: All features work exactly the same - no API changes
5. **Benefits**: Better code organization, easier maintenance, improved testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes to the appropriate module
4. Test thoroughly (individual modules can be tested separately)
5. Submit a pull request

### Development Guidelines
- **Keep modules focused**: Each module should have a single responsibility
- **Update config.py**: Add new configuration constants to the centralized config
- **Test imports**: Ensure all modules can be imported independently
- **Follow structure**: Place new functionality in the appropriate module

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
3. Verify modular architecture integrity
4. Open an issue on GitHub
5. Include system specifications and error details

---

**Last Updated**: July 27, 2025  
**Version**: 2.0.0 (Modular Architecture)  
**GPU Optimized**: ✅ (AMD ROCm)  
**ZipEnhancer Integrated**: ✅  
**Architecture**: ✅ Modular (main.py entry point) 