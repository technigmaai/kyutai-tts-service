from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import sphn
import tempfile
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel
from pydub import AudioSegment
from pydub.effects import normalize
import uvicorn
import logging
import re
import os
import html
from xml.etree import ElementTree as ET

# Torch optimization and GPU setup (from simple.py)
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# GPU environment setup for AMD ROCm (from simple.py)
os.environ["MIOPEN_FIND_MODE"] = "FAST"
os.environ["MIOPEN_USER_DB_PATH"] = os.path.expanduser("./miopen_cache")
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# ZipEnhancer imports (from simple.py)
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.fileio import File
    import io
    ZIPENHANCER_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("ZipEnhancer noise suppression available")
except ImportError as e:
    ZIPENHANCER_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"ZipEnhancer not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="Kyutai TTS API", description="Generate downloadable speech from text.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Model Loading ---
MODEL_LOADED = False
tts_model = None
zipenhancer_pipeline = None

try:
    # Model loading configuration
    MODEL_REPO = "kyutai/tts-1.6b-en_fr"
    VOICE_REPO = "kyutai/tts-voices"
    DEFAULT_VOICE = "Happy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
    # Instantiate the TTS model
    tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=device)
    tts_model.voice_repo = VOICE_REPO
    MODEL_LOADED = True
    logger.info("TTS Model loaded successfully.")
    
    # Initialize ZipEnhancer pipeline (from simple.py)
    if ZIPENHANCER_AVAILABLE:
        try:
            zipenhancer_pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model='iic/speech_zipenhancer_ans_multiloss_16k_base'
            )
            logger.info("ZipEnhancer pipeline loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load ZipEnhancer pipeline: {e}")
            zipenhancer_pipeline = None
            ZIPENHANCER_AVAILABLE = False
    
except Exception as e:
    MODEL_LOADED = False
    logger.exception(f"FATAL: Error loading model: {e}")
    # Define a placeholder if loading fails
    def tts_model_placeholder():
        raise RuntimeError("TTS Model could not be loaded. The application cannot proceed.")
    tts_model = tts_model_placeholder

# --- Voice and SSML Configuration ---
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

def parse_ssml(ssml_text: str, default_voice: str):
    """
    Parses an SSML string into a list of segments.
    Automatically sanitizes text content within SSML tags to handle special characters.
    """
    logger.info("Parsing input text...")
    segments = []
    stripped_text = ssml_text.strip()

    def escape_text_nodes(match):
        """Helper function for re.sub to escape only the text content."""
        return match.group(1) + html.escape(match.group(2)) + match.group(3)

    try:
        # Sanitize the text content *between* tags before parsing.
        sanitized_ssml = re.sub(r'(>)([^<>]*)(<)', escape_text_nodes, stripped_text)
        
        if not sanitized_ssml.startswith('<'):
             sanitized_ssml = html.escape(sanitized_ssml)

        root = ET.fromstring(sanitized_ssml)

    except ET.ParseError:
        # If parsing still fails, it's likely not SSML. Treat as plain text.
        logger.warning("SSML parsing failed even after sanitization. Treating input as plain text.")
        escaped_text = html.escape(stripped_text)
        ssml_for_text = f'<speak><voice name="{default_voice}">{escaped_text}</voice></speak>'
        root = ET.fromstring(ssml_for_text)

    # Process nodes directly under the <speak> tag
    for node in root:
        if node.tag == "voice":
            voice_name = node.attrib.get("name", default_voice)
            if voice_name not in VOICE_OPTIONS:
                logger.warning(f"Voice '{voice_name}' not supported. Using fallback '{default_voice}'.")
                voice_name = default_voice
            
            text = "".join(node.itertext()).strip()
            if text:
                segments.append((voice_name, text))
        
        elif node.tag == "break":
            duration_str = node.attrib.get("time", "500ms")
            match = re.match(r"(\d+)", duration_str)
            ms = int(match.group(1)) if match else 500
            silence = AudioSegment.silent(duration=ms)
            segments.append(("PAUSE", silence))
            
    logger.info(f"Parsed into {len(segments)} segments.")
    return segments

def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    Create WAV file header bytes for raw PCM data.
    
    :param dataflow: Audio bytes data
    :param sample_rate: Sample rate, default 16000
    :param num_channels: Number of channels, default 1 (mono)
    :param bits_per_sample: Bits per sample, default 16
    :return: Complete WAV file bytes with header
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # Build header with bytearray
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow

def apply_zipenhancer_windowed(audio_path: str, window_seconds: float = 2.0, 
                              zipenhancer_sample_rate: int = 16000, use_streaming: bool = True) -> str:
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
                                        zipenhancer_sample_rate=16000, use_streaming=True)
    else:
        # Simple processing (original method) with proper resampling
        if not ZIPENHANCER_AVAILABLE or not zipenhancer_pipeline:
            logger.warning("ZipEnhancer not available, skipping noise suppression")
            return audio_path
        
        try:
            logger.info("Applying ZipEnhancer noise suppression (simple mode with resampling)...")
            
            # Load and resample audio to 16kHz for ZipEnhancer
            audio = AudioSegment.from_wav(audio_path)
            resampled_audio = audio.set_frame_rate(16000).set_channels(1)
            
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
        attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=3.5)
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

    final_audio = final_audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    
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

# --- API Endpoint ---
class TTSRequest(BaseModel):
    text: str
    voice_choice: str = DEFAULT_VOICE
    output_format: str = "mp3"  # Output format: "mp3" or "wav"
    filename: str = None  # Custom output filename (without extension)
    apply_zipenhancer: bool = False  # Enable ZipEnhancer post-processing
    zipenhancer_quality: str = "high"  # Quality mode: "standard", "high", "ultra"
    zipenhancer_window_size: float = 2.0  # Window size in seconds for advanced processing
    # Audio processing parameters
    normalize: bool = False  # Apply audio normalization
    volume_boost: float = None  # Volume boost in decibels
    fade_in: int = None  # Fade-in duration in milliseconds
    fade_out: int = None  # Fade-out duration in milliseconds
    bitrate: str = "320k"  # MP3 bitrate (e.g., "128k", "192k", "320k")

@app.post("/api/tts")
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
            import re
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

@app.get("/api/zipenhancer/status")
def zipenhancer_status():
    """
    Returns the status of ZipEnhancer availability and configuration options.
    """
    return {
        "zipenhancer_available": ZIPENHANCER_AVAILABLE,
        "pipeline_loaded": zipenhancer_pipeline is not None,
        "quality_modes": {
            "standard": "Simple processing, fastest speed",
            "high": "Windowed processing, better quality (default)",
            "ultra": "Same as high with optimal settings"
        },
        "default_window_size": 2.0,
        "recommended_window_range": [1.0, 5.0]
    }

# --- Main Execution ---
if __name__ == "__main__":
    if MODEL_LOADED:
        logger.info("Launching FastAPI server on http://0.0.0.0:7861")
        uvicorn.run(app, host="0.0.0.0", port=7861)
    else:
        logger.error("Application will not start because the model failed to load.")
