# 🧪 Testing & Validation

This document describes how the Kyutai TTS Service is tested and validated,
what is covered, and how to run the test suites.

---

## Test Suites

Two scripts are provided in `scripts/`:

| Script | Coverage | Usage |
|--------|----------|-------|
| [`scripts/test.sh`](../scripts/test.sh) | Main suite — all API features, error handling, audio validation | `./scripts/test.sh [port...]` |
| [`scripts/test_edge.sh`](../scripts/test_edge.sh) | Edge cases — voices, filename, format, bitrate, window size | `./scripts/test_edge.sh [port]` |

Both scripts are **port-parameterized**, so they can validate any deployment:
bare-metal, Docker, or a remote instance.

```bash
# Test both default deployments (bare-metal 7861 + docker 7862)
./scripts/test.sh

# Test a single deployment
./scripts/test.sh 7861
./scripts/test_edge.sh 7862
```

**Requirements**: `curl`, `ffprobe` (static build in `~/bin`), `bc`.

---

## Main Suite Coverage (`scripts/test.sh`)

| # | Test | Validates |
|---|------|-----------|
| 1 | ZipEnhancer status | `zipenhancer_available` and `pipeline_loaded` are `true` |
| 2 | Basic TTS (MP3) | HTTP 200 + non-trivial audio size |
| 3 | Basic TTS (WAV) | HTTP 200 + non-trivial audio size |
| 4 | Custom filename | HTTP 200 + non-trivial audio size |
| 5 | SSML multi-voice | 3 voices + `<break>` pauses, HTTP 200 |
| 6 | ZipEnhancer standard | HTTP 200 + non-trivial audio size |
| 7 | ZipEnhancer high (WAV) | HTTP 200 + non-trivial audio size |
| 8 | ZipEnhancer ultra | HTTP 200 + non-trivial audio size |
| 9 | Audio effects | normalize + volume boost + fades + bitrate |
| 10 | Complete pipeline | ZipEnhancer + effects + custom filename combined |
| 11 | Invalid voice | Falls back to default voice (HTTP 200, not 500) |
| 12 | Empty text | Handled gracefully (HTTP 200 or 400) |
| 13 | Audio validation | `ffprobe` confirms valid format + duration > 0.5s |
| 14 | SSML pause correctness | Duration ≥ 3s (1s + 2s breaks actually applied) |

## Edge Suite Coverage (`scripts/test_edge.sh`)

| # | Test | Validates |
|---|------|-----------|
| 1 | All 23 voices | Every voice in `VOICE_OPTIONS` generates valid audio |
| 2 | Custom filename header | `Content-Disposition` contains the requested filename |
| 3 | Invalid output format | `ogg` falls back to MP3 (`audio/mpeg`) |
| 4 | Bitrate | Requested `128k` is actually applied (measured via ffprobe) |
| 5 | ZipEnhancer window size | Upper bound (5.0s) works |

---

## Validation Approach

- **HTTP status codes** — every request must return `200` (or a documented error code).
- **Audio size** — generated files must be non-trivial (> 1KB), proving real audio was produced.
- **`ffprobe`** — independently verifies the output is a valid audio file with the
  expected format (MP3/WAV) and a plausible duration.
- **SSML pause check** — the multi-voice test asserts the total duration includes
  the requested `<break>` pauses, guarding against the SSML parsing bug fixed in
  this project (where `time="1s"` was parsed as 1ms).
- **Error-path checks** — invalid voice and empty text must not crash the service.

---

## Known Issues Found by These Tests

The test suite caught a real bug during development:

- **Invalid `voice_choice` → HTTP 500** (`KeyError: 'NotARealVoice'`). An invalid
  voice propagated as `default_voice` into `generate_audio()`, where
  `VOICE_OPTIONS[default_voice]` raised `KeyError`. Fixed by validating
  `default_voice` against `VOICE_OPTIONS` at the top of `generate_audio()` and
  falling back to the configured `DEFAULT_VOICE`.

---

## Performance Baseline

Measured on EVO-X2 (AMD Strix Halo gfx1151, ROCm 7.1.4 / container ROCm 7.2.4):

| Deployment | Avg time per TTS request |
|-----------|--------------------------|
| Bare-metal (port 7861) | ~5.3s |
| Docker (port 7862) | ~5.5s |

---

## Related

- [Back to README](../README.md)
- [API Reference](API.md)
- [Troubleshooting](TROUBLESHOOTING.md)
