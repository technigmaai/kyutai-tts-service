# 🐛 Troubleshooting

Common issues and how to resolve them.

---

## 1. GPU Memory Issues

```bash
# Check GPU availability (must set LD_LIBRARY_PATH first)
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If the GPU is not detected, verify:
- ROCm is installed (`ls /opt/rocm`)
- `LD_LIBRARY_PATH` includes the ROCm lib dirs
- The GPU is supported (gfx1151 works natively with the AMD-built wheels)

## 2. Slow Performance

```bash
# Use standard mode for faster processing
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "apply_zipenhancer": true, "zipenhancer_quality": "standard"}'
```

## 3. Audio Quality Issues

```bash
# Enable ZipEnhancer for better quality
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "apply_zipenhancer": true, "zipenhancer_quality": "high"}'
```

## 4. Service Not Starting

```bash
# Check if all modular components are present
.venv/bin/python -c "from main import main; print('✅ Modular architecture working')"

# Check GPU availability (must set LD_LIBRARY_PATH first)
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"

# Check dependencies
.venv/bin/python -m pip list | grep -i torch

# Check the service log
cat /tmp/service.log
```

> **Note**: `HSA_OVERRIDE_GFX_VERSION` is **no longer needed**. The AMD-built
> PyTorch wheels support gfx1151 (Strix Halo) natively.

## 5. `HIPBLAS_STATUS_INVALID_VALUE` / `hipblasLtMatmulAlgoGetHeuristic` error

This happens when the AMD-built PyTorch wheel's hipBLASLt path is used against
system ROCm 7.1.4 (missing `TensileLibrary_lazy_gfx1100.dat`). The service
already handles this by forcing the plain hipBLAS backend in
`tts/engine.py::initialize_environment()`. If you see this error, verify the
backend is set:

```bash
.venv/bin/python -c "import torch; print(torch.backends.cuda.preferred_blas_library())"
# Should print: _BlasBackend.Hipblas
```

## 6. Import Errors (Modular Architecture)

```bash
# Verify all modules can be imported
.venv/bin/python -c "
import config
from utils.ssml_parser import parse_ssml
from audio.processing import initialize_zipenhancer
from tts.engine import initialize_environment
from api.models import TTSRequest
from api.routes import router
print('✅ All modules imported successfully')
"
```

---

## Log Analysis

The service provides detailed logging:
- **GPU Usage**: Real-time memory and utilization
- **Processing Time**: Per-segment and total timing
- **File Operations**: Temporary file creation and cleanup
- **Error Tracking**: Detailed error messages and stack traces
- **Module Loading**: Component initialization status

Logs go to `/tmp/service.log` when started via `./restart.sh`, or to the
terminal when started via `./start.sh`.

---

## Related

- [Back to README](../README.md)
- [Installation](INSTALLATION.md)
- [API Reference](API.md)
