# Kyutai TTS Service - Docker image
#
# Base: ROCm 7.2.4 (Ubuntu 24.04) - matches the AMD-built PyTorch wheels
# exactly, which avoids the hipBLASLt version-mismatch seen on the host
# (host runs ROCm 7.1.4, wheels are built for 7.2.4).
#
# Build:
#   docker build -t kyutai-tts:latest .
#
# Run (ROCm GPU passthrough):
#   docker run -d --rm \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --security-opt seccomp=unconfined \
#     -p 7861:7861 -v kyutai-models:/models \
#     kyutai-tts:latest
#
# Or use docker-compose: docker compose up -d --build

FROM rocm/dev-ubuntu-24.04:7.2.4

# --- Base tools -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        gpg \
        wget \
    && rm -rf /var/lib/apt/lists/*

# --- ROCm compute libraries ------------------------------------------------
# The slim rocm/dev image ships only the core runtime; torch needs MIOpen,
# rocBLAS and hipBLASLt. Install them from the matching ROCm 7.2.4 apt repo.
RUN wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor \
        -o /usr/share/keyrings/rocm-keyring.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main' \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        rocblas \
        rocsolver \
        rocsparse \
        hipblas \
        hipblaslt \
        miopen-hip \
        hip-runtime-amd \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# --- uv + Python 3.13 (AMD ROCm wheels are cp310-cp313 only) --------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# --- Python deps (requirements first for layer caching) -------------------
COPY requirements.txt .
RUN uv venv --python 3.13 .venv

# AMD-built PyTorch stack (native gfx1151 support, matches ROCm 7.2.4)
RUN VIRTUAL_ENV=.venv uv pip install \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.9.1%2Brocm7.2.4.lw.git39497456-cp313-cp313-linux_x86_64.whl \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.5.1%2Brocm7.2.4.gita272dfa8-cp313-cp313-linux_x86_64.whl \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchaudio-2.9.0%2Brocm7.2.4.gite3c6ee2b-cp313-cp313-linux_x86_64.whl \
        https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torchvision-0.24.0%2Brocm7.2.4.gitb919bd0c-cp313-cp313-linux_x86_64.whl \
    && VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# --- ffmpeg (static build) for pydub MP3 export ---------------------------
RUN mkdir -p /usr/local/bin \
    && curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ -C /tmp \
    && cp /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg-*

# --- Application ----------------------------------------------------------
COPY . .

# Model caches live in a persistent volume (downloaded once, reused)
ENV HF_HOME=/models \
    MODELSCOPE_CACHE=/models/modelscope \
    MIOPEN_USER_DB_PATH=/models/miopen
VOLUME /models

EXPOSE 7861

CMD [".venv/bin/python", "main.py"]
