"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel
from typing import Optional
from config import (
    DEFAULT_VOICE, DEFAULT_OUTPUT_FORMAT, DEFAULT_BITRATE,
    DEFAULT_ZIPENHANCER_QUALITY, DEFAULT_ZIPENHANCER_WINDOW_SIZE
)

class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str
    voice_choice: str = DEFAULT_VOICE
    output_format: str = DEFAULT_OUTPUT_FORMAT  # Output format: "mp3" or "wav"
    filename: Optional[str] = None  # Custom output filename (without extension)
    apply_zipenhancer: bool = False  # Enable ZipEnhancer post-processing
    zipenhancer_quality: str = DEFAULT_ZIPENHANCER_QUALITY  # Quality mode: "standard", "high", "ultra"
    zipenhancer_window_size: float = DEFAULT_ZIPENHANCER_WINDOW_SIZE  # Window size in seconds for advanced processing
    # Audio processing parameters
    normalize: bool = False  # Apply audio normalization
    volume_boost: Optional[float] = None  # Volume boost in decibels
    fade_in: Optional[int] = None  # Fade-in duration in milliseconds
    fade_out: Optional[int] = None  # Fade-out duration in milliseconds
    bitrate: str = DEFAULT_BITRATE  # MP3 bitrate (e.g., "128k", "192k", "320k")

class ZipEnhancerStatusResponse(BaseModel):
    """Response model for ZipEnhancer status endpoint"""
    zipenhancer_available: bool
    pipeline_loaded: bool
    quality_modes: dict
    default_window_size: float
    recommended_window_range: list

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str 