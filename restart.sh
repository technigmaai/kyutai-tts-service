#!/bin/bash
# Restart the Kyutai TTS service.
# Stops any running instance, then starts a fresh one in the background.
# Logs go to /tmp/service.log

cd "$(dirname "$0")"

# Stop any running instance
pkill -f "python main.py" 2>/dev/null
sleep 2

# ROCm library path required by AMD-built PyTorch wheels (libroctx64.so.4 etc.)
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Add user-local bin (static ffmpeg for pydub MP3 export) to PATH
if [ -d "$HOME/bin" ]; then
    export PATH="$HOME/bin:$PATH"
fi

# Start the service detached from the terminal
setsid bash -c "exec .venv/bin/python main.py > /tmp/service.log 2>&1" < /dev/null &

echo "🚀 Service restarting..."
echo "📡 http://localhost:7861  |  logs: /tmp/service.log"
