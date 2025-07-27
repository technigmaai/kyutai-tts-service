"""
Audio processing utilities including ZipEnhancer noise suppression and audio effects
"""

import logging
import tempfile
import os
from pydub import AudioSegment
from pydub.effects import normalize
from utils.ssml_parser import create_wav_header
from config import ZIPENHANCER_SAMPLE_RATE

logger = logging.getLogger(__name__)

# ZipEnhancer imports and initialization
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.fileio import File
    import io
    ZIPENHANCER_AVAILABLE = True
    logger.info("ZipEnhancer noise suppression available")
except ImportError as e:
    ZIPENHANCER_AVAILABLE = False
    logger.warning(f"ZipEnhancer not available: {e}")

# Global pipeline variable - will be initialized by main
zipenhancer_pipeline = None

def initialize_zipenhancer():
    """Initialize ZipEnhancer pipeline"""
    global zipenhancer_pipeline
    
    if ZIPENHANCER_AVAILABLE:
        try:
            zipenhancer_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model='iic/speech_zipenhancer_ans_multiloss_16k_base'
            )
            logger.info("ZipEnhancer pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ZipEnhancer pipeline: {e}")
            zipenhancer_pipeline = None
            return False
    return False

def apply_zipenhancer_windowed(audio_path: str, window_seconds: float = 2.0, 
                              zipenhancer_sample_rate: int = ZIPENHANCER_SAMPLE_RATE, 
                              use_streaming: bool = True) -> str:
    """
    Apply ZipEnhancer noise suppression with advanced windowed processing.
    This provides better quality and memory efficiency for large audio files.
    
    :param audio_path: Path to input audio file
    :param window_seconds: Window size in seconds for streaming processing
    :param zipenhancer_sample_rate: ZipEnhancer target sample rate (16kHz)
    :param use_streaming: Whether to use windowed streaming processing
    :return: Path to enhanced audio file
    """
    if not ZIPENHANCER_AVAILABLE or not zipenhancer_pipeline:
        logger.warning("ZipEnhancer not available, skipping noise suppression")
        return audio_path
    
    try:
        logger.info(f"Applying ZipEnhancer with windowed processing (window: {window_seconds}s, streaming: {use_streaming})")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as enhanced_fp:
            enhanced_path = enhanced_fp.name
        
        if not use_streaming:
            # Simple processing (original method) - resample first
            logger.info("Using simple processing with resampling...")
            
            # Load and resample audio to 16kHz for ZipEnhancer
            audio = AudioSegment.from_wav(audio_path)
            resampled_audio = audio.set_frame_rate(zipenhancer_sample_rate).set_channels(1)
            
            # Create temporary resampled file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_resampled:
                resampled_audio.export(temp_resampled.name, format="wav")
                temp_resampled_path = temp_resampled.name
            
            try:
                result = zipenhancer_pipeline(temp_resampled_path, output_path=enhanced_path)
                logger.info(f"ZipEnhancer simple processing completed: {enhanced_path}")
                os.remove(temp_resampled_path)
                return enhanced_path
            except Exception as e:
                logger.error(f"Simple processing failed: {e}")
                os.remove(temp_resampled_path)
                return audio_path
        
        # Advanced windowed processing
        logger.info("Using advanced windowed processing with proper resampling...")
        
        # Load and resample audio to 16kHz for ZipEnhancer
        audio = AudioSegment.from_wav(audio_path)
        resampled_audio = audio.set_frame_rate(zipenhancer_sample_rate).set_channels(1)
        
        # Check if audio is shorter than window size - if so, use simple processing
        audio_duration_seconds = len(resampled_audio) / 1000.0  # pydub uses milliseconds
        if audio_duration_seconds <= window_seconds:
            logger.info(f"Audio duration ({audio_duration_seconds:.2f}s) <= window size ({window_seconds}s), using simple processing instead")
            
            # Create temporary resampled file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_resampled:
                resampled_audio.export(temp_resampled.name, format="wav")
                temp_resampled_path = temp_resampled.name
            
            try:
                result = zipenhancer_pipeline(temp_resampled_path, output_path=enhanced_path)
                logger.info(f"ZipEnhancer simple processing completed: {enhanced_path}")
                os.remove(temp_resampled_path)
                return enhanced_path
            except Exception as e:
                logger.error(f"Simple processing failed: {e}")
                os.remove(temp_resampled_path)
                return audio_path
        
        # Export resampled audio to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_resampled:
            resampled_audio.export(temp_resampled.name, format="wav")
            temp_resampled_path = temp_resampled.name
        
        # Calculate window size in bytes for 16kHz audio (as in original simple.py)
        window_size = int(window_seconds * zipenhancer_sample_rate * 2)  # 2 bytes per sample for 16-bit
        logger.info(f"Window size: {window_size} bytes for {window_seconds}s at {zipenhancer_sample_rate}Hz")
        
        outputs = b''
        total_bytes_len = 0
        window_count = 0
        
        with open(temp_resampled_path, 'rb') as audiostream:
            # Skip WAV header (44 bytes)
            wav_header = audiostream.read(44)
            
            # Process audio in windows
            for dataflow in iter(lambda: audiostream.read(window_size), b''):
                if len(dataflow) == 0:
                    break
                    
                total_bytes_len += len(dataflow)
                window_count += 1
                
                logger.debug(f"Processing window {window_count}, size: {len(dataflow)} bytes")
                
                # Create WAV header for this chunk
                wav_chunk = create_wav_header(dataflow, sample_rate=zipenhancer_sample_rate, 
                                            num_channels=1, bits_per_sample=16)
                
                # Process chunk with ZipEnhancer
                result = zipenhancer_pipeline(wav_chunk)
                output_pcm = result['output_pcm']
                outputs += output_pcm
        
        # Clean up temporary resampled file
        os.remove(temp_resampled_path)
        
        # Trim output to original length
        outputs = outputs[:total_bytes_len]
        
        # Create temporary 16kHz enhanced file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_enhanced_16k:
            final_wav_16k = create_wav_header(outputs, sample_rate=zipenhancer_sample_rate, 
                                            num_channels=1, bits_per_sample=16)
            temp_enhanced_16k.write(final_wav_16k)
            temp_enhanced_16k_path = temp_enhanced_16k.name
        
        # Resample back to 44.1kHz to match original TTS output
        enhanced_audio_16k = AudioSegment.from_wav(temp_enhanced_16k_path)
        enhanced_audio_44k = enhanced_audio_16k.set_frame_rate(44100)
        enhanced_audio_44k.export(enhanced_path, format="wav")
        
        # Clean up temporary 16kHz file
        os.remove(temp_enhanced_16k_path)
        
        logger.info(f"ZipEnhancer windowed processing completed: {enhanced_path} ({window_count} windows)")
        return enhanced_path
        
    except Exception as e:
        logger.error(f"ZipEnhancer windowed processing failed: {e}")
        logger.info("Falling back to simple processing...")
        
        # Fallback to simple processing
        try:
            # Load and resample audio to 16kHz for ZipEnhancer
            audio = AudioSegment.from_wav(audio_path)
            resampled_audio = audio.set_frame_rate(zipenhancer_sample_rate).set_channels(1)
            
            # Create temporary resampled file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_resampled:
                resampled_audio.export(temp_resampled.name, format="wav")
                temp_resampled_path = temp_resampled.name
            
            result = zipenhancer_pipeline(temp_resampled_path, output_path=enhanced_path)
            logger.info(f"ZipEnhancer fallback processing completed: {enhanced_path}")
            os.remove(temp_resampled_path)
            return enhanced_path
        except Exception as fallback_error:
            logger.error(f"ZipEnhancer fallback also failed: {fallback_error}")
            return audio_path  # Return original if all enhancement fails

def apply_zipenhancer(audio_path: str, use_advanced_processing: bool = True, 
                     window_seconds: float = 2.0) -> str:
    """
    Apply ZipEnhancer noise suppression to audio file.
    
    :param audio_path: Path to input audio file
    :param use_advanced_processing: Whether to use windowed processing for better quality
    :param window_seconds: Window size for advanced processing
    :return: Path to enhanced audio file
    """
    if use_advanced_processing:
        return apply_zipenhancer_windowed(audio_path, window_seconds=window_seconds, 
                                        zipenhancer_sample_rate=ZIPENHANCER_SAMPLE_RATE, use_streaming=True)
    else:
        # Simple processing (original method) with proper resampling
        if not ZIPENHANCER_AVAILABLE or not zipenhancer_pipeline:
            logger.warning("ZipEnhancer not available, skipping noise suppression")
            return audio_path
        
        try:
            logger.info("Applying ZipEnhancer noise suppression (simple mode with resampling)...")
            
            # Load and resample audio to 16kHz for ZipEnhancer
            audio = AudioSegment.from_wav(audio_path)
            resampled_audio = audio.set_frame_rate(ZIPENHANCER_SAMPLE_RATE).set_channels(1)
            
            # Create temporary resampled file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_resampled:
                resampled_audio.export(temp_resampled.name, format="wav")
                temp_resampled_path = temp_resampled.name
            
            # Create output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as enhanced_fp:
                enhanced_path = enhanced_fp.name
            
            try:
                result = zipenhancer_pipeline(temp_resampled_path, output_path=enhanced_path)
                logger.info(f"ZipEnhancer simple processing completed: {enhanced_path}")
                os.remove(temp_resampled_path)
                return enhanced_path
            except Exception as e:
                logger.error(f"ZipEnhancer processing failed: {e}")
                os.remove(temp_resampled_path)
                return audio_path
            
        except Exception as e:
            logger.error(f"ZipEnhancer processing failed: {e}")
            return audio_path

def apply_audio_effects(audio: AudioSegment, normalize_audio: bool = False, 
                       volume_boost: float = None, fade_in_ms: int = None, 
                       fade_out_ms: int = None) -> AudioSegment:
    """
    Apply audio processing effects to an AudioSegment.
    
    :param audio: Input AudioSegment
    :param normalize_audio: Apply normalization
    :param volume_boost: Volume boost in decibels
    :param fade_in_ms: Fade-in duration in milliseconds
    :param fade_out_ms: Fade-out duration in milliseconds
    :return: Processed AudioSegment
    """
    try:
        processed_audio = audio
        
        # Apply normalization
        if normalize_audio:
            logger.info("Applying audio normalization")
            processed_audio = normalize(processed_audio)
        
        # Apply volume boost
        if volume_boost is not None:
            logger.info(f"Applying volume boost: {volume_boost} dB")
            processed_audio = processed_audio.apply_gain(volume_boost)
        
        # Apply fade-in
        if fade_in_ms is not None and fade_in_ms > 0:
            logger.info(f"Applying fade-in: {fade_in_ms} ms")
            processed_audio = processed_audio.fade_in(fade_in_ms)
        
        # Apply fade-out
        if fade_out_ms is not None and fade_out_ms > 0:
            logger.info(f"Applying fade-out: {fade_out_ms} ms")
            processed_audio = processed_audio.fade_out(fade_out_ms)
        
        return processed_audio
        
    except Exception as e:
        logger.error(f"Error applying audio effects: {e}")
        return audio  # Return original audio if effects fail 