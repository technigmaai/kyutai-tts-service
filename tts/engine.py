"""
TTS Engine module for Moshi TTS model loading and audio generation
"""

import logging
import torch
import numpy as np
import sphn
import tempfile
import os
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel
from pydub import AudioSegment
from utils.ssml_parser import parse_ssml
from audio.processing import apply_zipenhancer, apply_audio_effects
from config import (
    MODEL_REPO, VOICE_REPO, DEFAULT_VOICE, VOICE_OPTIONS, 
    DEFAULT_SAMPLE_RATE, GPU_ENV_VARS, TORCH_NUM_THREADS, 
    TORCH_NUM_INTEROP_THREADS
)

logger = logging.getLogger(__name__)

# Global variables for TTS model
MODEL_LOADED = False
tts_model = None

def initialize_environment():
    """Initialize GPU environment and torch settings"""
    # Torch optimization and GPU setup
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

    # GPU environment setup for AMD ROCm
    for key, value in GPU_ENV_VARS.items():
        os.environ[key] = value

    # Use the plain hipBLAS backend instead of hipBLASLt.
    # The AMD-built PyTorch wheels (ROCm 7.2.4) call hipblasLtMatmulAlgoGetHeuristic,
    # which fails against the system ROCm 7.1.4 hipBLASLt with
    # HIPBLAS_STATUS_INVALID_VALUE (missing TensileLibrary_lazy_gfx1100.dat).
    # The non-LT hipBLAS path works correctly on gfx1151 (Strix Halo).
    try:
        torch.backends.cuda.preferred_blas_library("hipblas")
        logger.info("Using hipBLAS backend (hipBLASLt disabled for ROCm compatibility)")
    except Exception as e:
        logger.warning(f"Could not set preferred BLAS backend to hipblas: {e}")

def initialize_tts_model():
    """Initialize the TTS model"""
    global MODEL_LOADED, tts_model
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
        # Instantiate the TTS model
        tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=device)
        tts_model.voice_repo = VOICE_REPO
        MODEL_LOADED = True
        logger.info("TTS Model loaded successfully.")
        return True
        
    except Exception as e:
        MODEL_LOADED = False
        logger.exception(f"FATAL: Error loading TTS model: {e}")
        # Define a placeholder if loading fails
        def tts_model_placeholder():
            raise RuntimeError("TTS Model could not be loaded. The application cannot proceed.")
        tts_model = tts_model_placeholder
        return False

def generate_audio(ssml_text: str, default_voice: str, output_format: str = "mp3",
                  apply_zipenhancer_enhancement: bool = False, zipenhancer_quality: str = "high", 
                  zipenhancer_window_size: float = 2.0, normalize_audio: bool = False,
                  volume_boost: float = None, fade_in_ms: int = None, fade_out_ms: int = None,
                  bitrate: str = "320k") -> str:
    """
    Generates an audio file from an SSML or plain text string in MP3 or WAV format.
    Optionally applies ZipEnhancer noise suppression as post-processing.
    """
    if not MODEL_LOADED:
        logger.error("Cannot generate audio: Model is not loaded.")
        raise RuntimeError("Model not loaded")

    logger.info("Generating audio from parsed segments...")
    segments = parse_ssml(ssml_text, default_voice)
    output_audio_segments = []

    for idx, (voice, content) in enumerate(segments):
        logger.info(f"Processing segment {idx + 1}/{len(segments)}: Voice='{voice}'")
        if voice == "PAUSE":
            output_audio_segments.append(content)
            continue

        entries = tts_model.prepare_script([content], padding_between=1)
        voice_path = tts_model.get_voice_path(VOICE_OPTIONS.get(voice, VOICE_OPTIONS[default_voice]))
        attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.5)
        result = tts_model.generate([entries], [attributes])

        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = [
                np.clip(tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()[0, 0], -1, 1)
                for frame in result.frames[tts_model.delay_steps:]
            ]
            pcm_data = np.concatenate(pcms, axis=-1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_fp:
            sphn.write_wav(wav_fp.name, pcm_data, tts_model.mimi.sample_rate)
            wav_path = wav_fp.name

        segment_audio = AudioSegment.from_wav(wav_path)
        output_audio_segments.append(segment_audio)
        os.remove(wav_path)

    if not output_audio_segments:
        logger.warning("No audio was generated, returning empty audio file.")
        final_audio = AudioSegment.silent(duration=1)
    else:
        final_audio = sum(output_audio_segments)

    final_audio = final_audio.set_frame_rate(DEFAULT_SAMPLE_RATE).set_sample_width(2).set_channels(1)
    
    # Export to temporary WAV for potential ZipEnhancer processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        final_audio.export(temp_wav.name, format="wav")
        temp_wav_path = temp_wav.name
    
    # Apply ZipEnhancer if requested
    if apply_zipenhancer_enhancement:
        logger.info("Applying ZipEnhancer post-processing...")
        
        # Determine processing mode based on quality setting
        use_advanced = zipenhancer_quality.lower() in ["high", "ultra"]
        
        enhanced_wav_path = apply_zipenhancer(
            temp_wav_path, 
            use_advanced_processing=use_advanced, 
            window_seconds=zipenhancer_window_size
        )
        
        # If enhancement was successful and created a new file, use it
        if enhanced_wav_path != temp_wav_path:
            os.remove(temp_wav_path)  # Remove original temp file
            temp_wav_path = enhanced_wav_path
    
    # Load the final audio (potentially enhanced) and apply audio effects
    final_enhanced_audio = AudioSegment.from_wav(temp_wav_path)
    
    # Apply audio processing effects
    processed_audio = apply_audio_effects(
        final_enhanced_audio,
        normalize_audio=normalize_audio,
        volume_boost=volume_boost,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms
    )
    
    # Validate output format
    if output_format.lower() not in ["mp3", "wav"]:
        logger.warning(f"Invalid output format '{output_format}', defaulting to MP3")
        output_format = "mp3"
    
    output_format = output_format.lower()
    file_suffix = f".{output_format}"
    
    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as output_fp:
        if output_format == "mp3":
            processed_audio.export(output_fp.name, format="mp3", bitrate=bitrate)
        else:  # wav
            processed_audio.export(output_fp.name, format="wav")
        
        logger.info(f"Final audio successfully exported to {output_fp.name} (format: {output_format.upper()})")
        
        # Cleanup temporary WAV file
        os.remove(temp_wav_path)
        
        return output_fp.name 