# ⚙️ Configuration

All configuration is centralized in `config.py`. This document covers the
available settings, performance options, ZipEnhancer quality modes, and SSML
advanced usage.

---

## Performance Settings

The service is pre-configured for optimal performance with:
- **GPU Acceleration**: Automatic ROCm utilization for AMD GPUs (native gfx1151)
- **hipBLAS backend**: Uses plain hipBLAS instead of hipBLASLt (the AMD wheel's
  hipBLASLt path is incompatible with system ROCm 7.1.4 — see
  [Troubleshooting](TROUBLESHOOTING.md))
- **Memory Management**: Automatic temporary file cleanup
- **Torch Optimization**: 8 threads for CPU operations, optimized interop threads

---

## `config.py` Settings

### Model Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `kyutai/tts-1.6b-en_fr` | HuggingFace repo for the TTS model |
| `VOICE_REPO` | `kyutai/tts-voices` | HuggingFace repo for voice samples |
| `DEFAULT_VOICE` | `"Happy"` | Default voice when none is specified |
| `VOICE_OPTIONS` | dict | 20+ named voices mapped to sample files |

### GPU Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `GPU_ENV_VARS` | `{"MIOPEN_FIND_MODE": "FAST", "MIOPEN_USER_DB_PATH": "~/.cache/miopen"}` | ROCm env vars set at startup |
| `ROCM_LIB_PATHS` | `["/opt/rocm/core-7.14/lib", "/opt/rocm/lib"]` | ROCm library paths (used by `start.sh`) |

> `HSA_OVERRIDE_GFX_VERSION` is intentionally **not** set — gfx1151 is natively
> supported by ROCm 7.x and the AMD-built PyTorch wheels.

### Torch Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `TORCH_NUM_THREADS` | `8` | CPU threads for torch operations |
| `TORCH_NUM_INTEROP_THREADS` | `8` | Inter-op parallelism threads |

### Audio Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_OUTPUT_FORMAT` | `"mp3"` | Default output format |
| `DEFAULT_BITRATE` | `"320k"` | Default MP3 bitrate |
| `DEFAULT_SAMPLE_RATE` | `44100` | Output sample rate (Hz) |

### ZipEnhancer Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_ZIPENHANCER_QUALITY` | `"high"` | Default quality mode |
| `DEFAULT_ZIPENHANCER_WINDOW_SIZE` | `2.0` | Default window size (seconds) |
| `ZIPENHANCER_SAMPLE_RATE` | `16000` | ZipEnhancer processing sample rate (Hz) |

### Server Settings
| Constant | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `"0.0.0.0"` | Bind address |
| `SERVER_PORT` | `7861` | Port |

---

## ZipEnhancer Quality Modes

The service offers three ZipEnhancer processing modes:

### Standard Mode (`zipenhancer_quality: "standard"`)
- **Processing**: Simple, single-pass enhancement
- **Speed**: Fastest processing time
- **Memory**: Low memory usage
- **Best for**: Quick processing, shorter audio files

### High Mode (`zipenhancer_quality: "high"`) — **Default**
- **Processing**: Windowed processing with 2-second chunks
- **Speed**: Moderate processing time
- **Memory**: Efficient memory usage for large files
- **Quality**: Better noise suppression and artifact removal
- **Best for**: Most use cases, balanced quality/speed

### Ultra Mode (`zipenhancer_quality: "ultra"`)
- **Processing**: Same as High with optimized settings
- **Speed**: Similar to High mode
- **Quality**: Maximum noise suppression quality
- **Best for**: Professional audio, critical applications

### Window Size Configuration
- **Default**: 2.0 seconds per processing window
- **Range**: 1.0 – 5.0 seconds recommended
- **Smaller windows**: Better quality, more processing time
- **Larger windows**: Faster processing, potential quality trade-offs

---

## SSML Advanced Usage

The service supports Speech Synthesis Markup Language (SSML) for advanced voice
control:

```xml
<speak>
  <voice name="Happy">Hello! This is a happy voice.</voice>
  <break time="1s"/>
  <voice name="Sad">This is a sad voice.</voice>
  <break time="2s"/>
  <voice name="Angry">And this is an angry voice!</voice>
</speak>
```

### Supported SSML Elements
- **`<voice name="...">`** — switch voice for the enclosed text (must be a valid
  key from `VOICE_OPTIONS`; unknown voices fall back to the default).
- **`<break time="..."/>`** — insert a pause. Supports `ms` and `s` units
  (e.g. `500ms`, `1s`, `1.5s`). Defaults to `500ms`.

### SSML Processing Flow
For multiple text segments, the service automatically:
1. Parses SSML into voice segments and pauses (handled by `utils/ssml_parser.py`)
2. Processes each segment with the specified voice (handled by `tts/engine.py`)
3. Reconstructs audio in original order
4. Applies ZipEnhancer if requested (handled by `audio/processing.py`)

---

## Related

- [Back to README](../README.md)
- [API Reference](API.md)
- [Troubleshooting](TROUBLESHOOTING.md)
