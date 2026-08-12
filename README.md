# 🎵 GPU-Optimized Text-to-Speech Service

A high-performance, GPU-accelerated Text-to-Speech (TTS) service built with
FastAPI, PyTorch, and Moshi TTS. Features ZipEnhancer noise suppression, SSML
support for multiple voices, and optimized processing for AMD GPUs with ROCm.

> **✅ Tested Hardware**: AMD Strix Halo (Ryzen AI Max+ 395) GPU (GFX1151 / Radeon 8060S) with ROCm 7.1.4

## 🧰 Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| **OS** | Ubuntu 26.04 LTS | tested on EVO-X2 |
| **ROCm** | 7.1.4 | system-wide, native gfx1151 support |
| **Python** | 3.13 | required — AMD ROCm wheels are cp310–cp313 only |
| **PyTorch** | 2.9.1+rocm7.2.4 | AMD-built wheel (repo.radeon.com), native gfx1151 |
| **torchaudio** | 2.9.0+rocm7.2.4 | AMD-built wheel |
| **torchvision** | 0.24.0+rocm7.2.4 | AMD-built wheel |
| **triton** | 3.5.1+rocm7.2.4 | AMD-built wheel |
| **moshi** | 0.2.13 | TTS engine (pins `torch<2.10`) |
| **modelscope** | 1.39.1 | ZipEnhancer noise suppression |
| **FastAPI** | 0.141.1 | web framework |
| **uvicorn** | 0.52.1 | ASGI server |
| **librosa** | 1.0.0 | audio analysis |
| **ffmpeg** | 7.0.2 (static) | pydub MP3 export (`~/bin`) |

## ✨ Features

- **GPU-Accelerated TTS** — AMD ROCm, native gfx1151 (no HSA override hack)
- **ZipEnhancer Noise Suppression** — windowed acoustic noise suppression
- **Audio Processing Effects** — normalization, volume boost, fade-in/out, bitrate
- **SSML Support** — multi-voice synthesis with `<voice>` and `<break>`
- **20+ Voice Options** — emotional and stylistic voices (Happy, Sad, Angry, Calm, …)
- **Flexible Output** — MP3/WAV with custom filenames and quality settings
- **Modular Design** — clean separation of concerns for easy maintenance

## 🚀 Quick Start

```bash
# 1. Install (see docs/INSTALLATION.md for full details)
./install.sh

# 2. Start the service (sets ROCm env, starts on port 7861)
./start.sh

# 3. Test it
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test!", "voice_choice": "Happy"}' \
  --output test_output.mp3
```

> Restart a running instance in the background with `./restart.sh`
> (logs to `/tmp/service.log`).

## 📚 Documentation

| Document | Contents |
|----------|----------|
| [**API Reference**](docs/API.md) | All endpoints, request parameters, and example requests |
| [**Installation**](docs/INSTALLATION.md) | Requirements, setup, GPU verification |
| [**Configuration**](docs/CONFIGURATION.md) | `config.py` settings, ZipEnhancer modes, SSML |
| [**Architecture**](docs/ARCHITECTURE.md) | Module breakdown, file structure, contributing |
| [**Troubleshooting**](docs/TROUBLESHOOTING.md) | Common issues and log analysis |

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙏 Acknowledgments

- **Moshi TTS**: Core TTS model (kyutai/tts-1.6b-en_fr)
- **PyTorch**: GPU acceleration framework
- **FastAPI**: Web framework
- **ROCm**: AMD GPU computing platform

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting guide](docs/TROUBLESHOOTING.md)
2. Review the logs for error messages
3. Verify modular architecture integrity
4. Open an issue on GitHub
5. Include system specifications and error details

---

**Last Updated**: August 2026  
**Version**: 2.1.0 (Modular Architecture, ROCm 7.x stack)  
**GPU Optimized**: ✅ (AMD ROCm 7.1.4, native gfx1151)  
**ZipEnhancer Integrated**: ✅  
**Architecture**: ✅ Modular (main.py entry point)
