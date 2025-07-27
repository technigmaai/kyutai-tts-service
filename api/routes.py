"""
FastAPI routes for the TTS service
"""

import logging
import re
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from api.models import TTSRequest, ZipEnhancerStatusResponse
from tts.engine import generate_audio
from audio.processing import ZIPENHANCER_AVAILABLE, zipenhancer_pipeline

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/api/tts")
def tts_endpoint(request: TTSRequest):
    """
    Handles TTS requests. Returns an audio file on success or a JSON error on failure.
    Now supports optional ZipEnhancer noise suppression post-processing.
    """
    logger.info("TTS API endpoint hit.")
    try:
        output_path = generate_audio(
            request.text, 
            request.voice_choice,
            output_format=request.output_format,
            apply_zipenhancer_enhancement=request.apply_zipenhancer,
            zipenhancer_quality=request.zipenhancer_quality,
            zipenhancer_window_size=request.zipenhancer_window_size,
            normalize_audio=request.normalize,
            volume_boost=request.volume_boost,
            fade_in_ms=request.fade_in,
            fade_out_ms=request.fade_out,
            bitrate=request.bitrate
        )
        
        # Set correct media type and filename based on output format
        if request.output_format.lower() == "wav":
            media_type = "audio/wav"
            default_filename = "generated_speech.wav"
            extension = ".wav"
        else:  # default to mp3
            media_type = "audio/mpeg"
            default_filename = "generated_speech.mp3"
            extension = ".mp3"
        
        # Use custom filename if provided, otherwise use default
        if request.filename:
            # Clean the filename of potentially harmful characters
            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', request.filename.strip())
            # Ensure it doesn't end with the extension already
            if safe_filename.lower().endswith(extension.lower()):
                final_filename = safe_filename
            else:
                final_filename = safe_filename + extension
        else:
            final_filename = default_filename
        
        return FileResponse(output_path, media_type=media_type, filename=final_filename)
    except ValueError as e:
        logger.error(f"Invalid input provided: {e}")
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        logger.exception("An internal error occurred during audio generation.")
        return JSONResponse(status_code=500, content={"detail": f"An internal error occurred: {str(e)}"})

@router.get("/api/zipenhancer/status", response_model=ZipEnhancerStatusResponse)
def zipenhancer_status():
    """
    Returns the status of ZipEnhancer availability and configuration options.
    """
    return ZipEnhancerStatusResponse(
        zipenhancer_available=ZIPENHANCER_AVAILABLE,
        pipeline_loaded=zipenhancer_pipeline is not None,
        quality_modes={
            "standard": "Simple processing, fastest speed",
            "high": "Windowed processing, better quality (default)",
            "ultra": "Same as high with optimal settings"
        },
        default_window_size=2.0,
        recommended_window_range=[1.0, 5.0]
    ) 