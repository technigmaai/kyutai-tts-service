# 📡 API Reference

The Kyutai TTS Service exposes a REST API on port `7861` (default). Interactive
documentation is available at `http://localhost:7861/docs` (Swagger UI).

- **Base URL**: `http://localhost:7861`
- **Content-Type**: `application/json`

---

## `POST /api/tts`

Generate speech from text with advanced options. Returns an audio file
(`audio/mpeg` for MP3, `audio/wav` for WAV) or a JSON error.

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **required** | Text to convert to speech (plain text or SSML) |
| `voice_choice` | string | `"Happy"` | Voice to use (Happy, Sad, Angry, etc.) |
| `output_format` | string | `"mp3"` | Output format: `"mp3"` or `"wav"` |
| `filename` | string | `null` | Custom output filename (extension auto-added) |
| `apply_zipenhancer` | boolean | `false` | Enable ZipEnhancer noise suppression post-processing |
| `zipenhancer_quality` | string | `"high"` | ZipEnhancer quality mode: `"standard"`, `"high"`, `"ultra"` |
| `zipenhancer_window_size` | float | `2.0` | Window size in seconds for windowed processing (1.0–5.0) |
| `normalize` | boolean | `false` | Apply audio normalization |
| `volume_boost` | float | `null` | Volume boost in decibels (e.g. `5.0`, `-3.0`) |
| `fade_in` | integer | `null` | Fade-in duration in milliseconds |
| `fade_out` | integer | `null` | Fade-out duration in milliseconds |
| `bitrate` | string | `"320k"` | MP3 bitrate (e.g. `"128k"`, `"192k"`, `"320k"`) |

### Example Requests

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

### Response

- **200 OK** — audio file (`audio/mpeg` or `audio/wav`), with `Content-Disposition` filename.
- **400 Bad Request** — invalid input (e.g. unsupported voice). Body: `{"detail": "..."}`.
- **500 Internal Server Error** — generation failure. Body: `{"detail": "..."}`.

---

## `GET /api/zipenhancer/status`

Returns the status of ZipEnhancer availability and configuration options.

```bash
curl "http://localhost:7861/api/zipenhancer/status"
```

### Response

```json
{
  "zipenhancer_available": true,
  "pipeline_loaded": true,
  "quality_modes": {
    "standard": "Simple processing, fastest speed",
    "high": "Windowed processing, better quality (default)",
    "ultra": "Same as high with optimal settings"
  },
  "default_window_size": 2.0,
  "recommended_window_range": [1.0, 5.0]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `zipenhancer_available` | boolean | Whether the modelscope ZipEnhancer package is installed |
| `pipeline_loaded` | boolean | Whether the noise-suppression pipeline is loaded in memory |
| `quality_modes` | object | Available quality modes and their descriptions |
| `default_window_size` | float | Default processing window in seconds |
| `recommended_window_range` | array | Recommended window size range `[min, max]` |

---

## Related

- [Back to README](../README.md)
- [Configuration & SSML](CONFIGURATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
