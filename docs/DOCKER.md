# 🐳 Docker Deployment (Parallel Option)

The service can run in a Docker container as an **alternative to the bare-metal
deployment**. Both can coexist on the same host — just use a different port for
one of them.

> **Why a container?** The image is based on `rocm/dev-ubuntu-24.04:7.2.4`,
> which **exactly matches** the AMD-built PyTorch wheels (built for ROCm 7.2.4).
> This avoids the hipBLASLt version-mismatch seen on hosts running a different
> ROCm version (e.g. 7.1.4), and makes the whole stack reproducible.
>
> The slim ROCm base image only ships the core runtime, so the Dockerfile also
> installs the compute libraries torch needs (rocBLAS, hipBLAS, hipBLASLt,
> MIOpen, hipFFT, hipRAND, hipSolver, hipSparse, RCCL) from the matching
> ROCm 7.2.4 apt repo.

---

## Build

```bash
docker build -t kyutai-tts:latest .
```

## Run (ROCm GPU passthrough)

```bash
docker run -d --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  -p 7861:7861 \
  -v kyutai-models:/models \
  kyutai-tts:latest
```

Or with Docker Compose (recommended):

```bash
docker compose up -d --build
```

### What the flags do
| Flag | Why |
|------|-----|
| `--device=/dev/kfd --device=/dev/dri` | Pass the AMD GPU devices into the container |
| `--group-add video` | Give the container access to the `video` group (GPU) |
| `--security-opt seccomp=unconfined` | Required for ROCm runtime in Docker |
| `-v kyutai-models:/models` | Persistent volume for the TTS + ZipEnhancer models (downloaded once) |

## Verify

```bash
# GPU visible inside the container?
docker exec kyutai-tts .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Service health
curl http://localhost:7861/api/zipenhancer/status
```

## Notes

- **Port conflict**: if the bare-metal service is already on `7861`, change the
  container's host port, e.g. `-p 7862:7861`.
- **Model cache**: the first startup downloads the ~3GB TTS model into the
  `kyutai-models` volume. Subsequent restarts are fast.
- **hipBLAS backend**: the code still prefers the plain hipBLAS backend. With
  matching ROCm 7.2.4 in the container this is not strictly required, but it is
  harmless and keeps behavior consistent across hosts.

---

## Related

- [Back to README](../README.md)
- [Installation (bare-metal)](INSTALLATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
