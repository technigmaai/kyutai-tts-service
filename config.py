"""
Configuration constants and default values for Kyutai TTS Service
"""

import os

# Model Configuration
MODEL_REPO = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"
DEFAULT_VOICE = "Happy"

# Voice Options Dictionary
VOICE_OPTIONS = {
    "Happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "Angry": "expresso/ex03-ex01_angry_001_channel1_201s.wav",
    "Awe": "expresso/ex03-ex01_awe_001_channel1_1323s.wav",
    "Calm": "expresso/ex03-ex01_calm_001_channel1_1143s.wav",
    "Confused": "expresso/ex03-ex01_confused_001_channel1_909s.wav",
    "Desire": "expresso/ex03-ex01_desire_004_channel1_545s.wav",
    "Disgusted": "expresso/ex03-ex01_disgusted_004_channel1_170s.wav",
    "Laughing": "expresso/ex03-ex01_laughing_001_channel1_188s.wav",
    "Sarcastic": "expresso/ex03-ex01_sarcastic_001_channel1_435s.wav",
    "Sleepy": "expresso/ex03-ex01_sleepy_001_channel1_619s.wav",
    "Sleepy2": "expresso/ex03-ex01_sleepy_001_channel2_662s.wav",
    "Default": "expresso/ex01-ex02_default_001_channel1_168s.wav",
    "Enunciated": "expresso/ex01-ex02_enunciated_001_channel1_432s.wav",
    "Fast": "expresso/ex01-ex02_fast_001_channel1_104s.wav",
    "Projected": "expresso/ex01-ex02_projected_001_channel1_46s.wav",
    "Whisper": "expresso/ex01-ex02_whisper_001_channel1_579s.wav",
    "Angry2": "expresso/ex03-ex01_angry_001_channel1_201s.wav",
    "Awe2": "expresso/ex03-ex01_awe_001_channel1_1323s.wav",
    "Desire2": "expresso/ex03-ex01_desire_004_channel1_545s.wav",
    "Sarcastic3": "expresso/ex03-ex01_sarcastic_001_channel1_435s.wav",
    "Child": "expresso/ex03-ex02_childdir-child_004_channel1_308s.wav",
    "Sarcastic2": "expresso/ex04-ex02_sarcastic_001_channel2_466s.wav",
    "Default2": "expresso/ex04-ex03_default_002_channel2_239s.wav"
}

# GPU and Environment Configuration
# These are set in the main service initialization
GPU_ENV_VARS = {
    "MIOPEN_FIND_MODE": "FAST",
    "MIOPEN_USER_DB_PATH": os.path.expanduser("~/Development/vscode/modelscope/miopen_cache"),
    "HSA_OVERRIDE_GFX_VERSION": "11.0.0"
}

# Torch Configuration
TORCH_NUM_THREADS = 8
TORCH_NUM_INTEROP_THREADS = 8

# Default Audio Settings
DEFAULT_OUTPUT_FORMAT = "mp3"
DEFAULT_BITRATE = "320k"
DEFAULT_SAMPLE_RATE = 44100

# ZipEnhancer Configuration
DEFAULT_ZIPENHANCER_QUALITY = "high"
DEFAULT_ZIPENHANCER_WINDOW_SIZE = 2.0
ZIPENHANCER_SAMPLE_RATE = 16000

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7861 