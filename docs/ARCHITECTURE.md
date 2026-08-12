# 🏗️ Architecture

The Kyutai TTS Service uses a **modular architecture** with clear separation of
concerns, making it easier to maintain, test, and extend.

---

## Module Overview

| Module | Responsibility |
|--------|----------------|
| `main.py` | Entry point, service orchestration, FastAPI app setup |
| `config.py` | Centralized configuration constants and defaults |
| `api/` | FastAPI routes and Pydantic request/response models |
| `tts/` | TTS model loading and audio generation |
| `audio/` | ZipEnhancer noise suppression and audio effects |
| `utils/` | SSML parsing and WAV header utilities |

---

## Module Details

### Entry Point (`main.py`)
- Service initialization and startup
- Model loading coordination
- FastAPI app configuration
- Error handling and logging setup

### Configuration (`config.py`)
- All configuration constants and defaults
- Voice options and model repositories
- GPU environment settings
- Performance optimizations

### API Layer (`api/`)
- **`models.py`**: Pydantic request/response models
- **`routes.py`**: FastAPI endpoint handlers and routing

### TTS Engine (`tts/engine.py`)
- TTS model loading and initialization
- Audio generation from text/SSML
- Voice processing and management
- GPU optimization settings (including the hipBLAS backend fix)

### Audio Processing (`audio/processing.py`)
- ZipEnhancer noise suppression
- Audio effects (normalization, volume, fades)
- Windowed processing for large files
- Audio format conversion

### Utilities (`utils/ssml_parser.py`)
- SSML parsing and validation
- WAV header creation for raw audio
- Text sanitization and processing

---

## File Structure

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
│   └── simple.py               # Original ZipEnhancer test
├── docs/                   # 📚 Documentation
│   ├── API.md             # 📡 API reference
│   ├── INSTALLATION.md    # 🛠️ Installation guide
│   ├── CONFIGURATION.md   # ⚙️ Configuration reference
│   ├── ARCHITECTURE.md    # 🏗️ This document
│   └── TROUBLESHOOTING.md # 🐛 Common issues
├── README.md               # 📖 Landing page
├── .gitignore             # 🚫 Git ignore rules
├── requirements.txt        # 📋 Python dependencies
├── install.sh             # 🔧 Automated installation (uv + AMD wheels)
├── start.sh               # ▶️ Quick start script (sets ROCm env, starts service)
└── restart.sh             # 🔄 Restart helper (stops + starts service in background)
```

> `.venv/` is created locally by `install.sh` and is git-ignored.

---

## Migration from Monolithic Version

If you're upgrading from the previous monolithic version:

1. **Backup**: Original files are automatically moved to `archive/`
2. **Configuration**: Settings are now centralized in `config.py`
3. **Startup**: Use `python main.py` or `./start.sh` instead of `kyutai-tts-service.py`
4. **Functionality**: All features work exactly the same — no API changes
5. **Benefits**: Better code organization, easier maintenance, improved testing

---

## Contributing

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

---

## Related

- [Back to README](../README.md)
- [Configuration & SSML](CONFIGURATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
