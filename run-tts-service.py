from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import sphn
import tempfile
import torch
import torch.nn.functional as F
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
from typing import Optional
import torchaudio
import torchaudio.transforms as T

# Audio cleaning imports - keeping CPU fallbacks
import librosa
from scipy import signal
from scipy.ndimage import median_filter
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="GPU-Optimized Kyutai TTS API", description="Generate and clean speech audio with GPU acceleration.", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- GPU-Accelerated Audio Cleaner Class ---
class GPUAudioCleaner:
    def __init__(self, audio_data, sample_rate, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to torch tensor and move to GPU
        if isinstance(audio_data, np.ndarray):
            self.audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
        else:
            self.audio_tensor = audio_data.float().to(self.device)
            
        self.sample_rate = sample_rate
        self.original_shape = self.audio_tensor.shape
        logger.debug(f"Audio tensor initialized on {self.device} with shape {self.original_shape}")
    
    def remove_dc_offset(self):
        """Remove DC offset using GPU operations"""
        self.audio_tensor = self.audio_tensor - torch.mean(self.audio_tensor)
        logger.debug("✓ DC offset removed (GPU)")
    
    def remove_clicks_pops_gpu(self, threshold=0.1):
        """Remove clicks and pops using GPU-accelerated median filtering"""
        # Detect sudden amplitude changes
        diff = torch.abs(torch.diff(self.audio_tensor))
        outliers = diff > threshold
        
        if torch.any(outliers):
            # GPU-based median filtering using conv1d
            kernel_size = 3
            padding = kernel_size // 2
            
            # Reshape for conv1d (batch, channels, length)
            audio_reshaped = self.audio_tensor.unsqueeze(0).unsqueeze(0)
            
            # Apply median-like smoothing using average pooling
            smoothed = F.avg_pool1d(audio_reshaped, kernel_size, stride=1, padding=padding)
            self.audio_tensor = smoothed.squeeze(0).squeeze(0)
            
            logger.debug(f"✓ Clicks and pops removed (GPU) - found {torch.sum(outliers).item()} outliers")
    
    def apply_highpass_filter_gpu(self, cutoff=80):
        """Apply high-pass filter using torchaudio on GPU"""
        try:
            # Create high-pass filter
            highpass = T.HighpassBiquad(
                sample_rate=self.sample_rate, 
                cutoff_freq=cutoff
            ).to(self.device)
            
            # Apply filter (need to add batch dimension)
            audio_filtered = highpass(self.audio_tensor.unsqueeze(0))
            self.audio_tensor = audio_filtered.squeeze(0)
            logger.debug(f"✓ High-pass filter applied (GPU, cutoff: {cutoff} Hz)")
        except Exception as e:
            logger.warning(f"GPU highpass failed, using CPU fallback: {e}")
            self._apply_highpass_filter_cpu(cutoff)
    
    def apply_lowpass_filter_gpu(self, cutoff=8000):
        """Apply low-pass filter using torchaudio on GPU"""
        try:
            # Create low-pass filter
            lowpass = T.LowpassBiquad(
                sample_rate=self.sample_rate, 
                cutoff_freq=cutoff
            ).to(self.device)
            
            # Apply filter
            audio_filtered = lowpass(self.audio_tensor.unsqueeze(0))
            self.audio_tensor = audio_filtered.squeeze(0)
            logger.debug(f"✓ Low-pass filter applied (GPU, cutoff: {cutoff} Hz)")
        except Exception as e:
            logger.warning(f"GPU lowpass failed, using CPU fallback: {e}")
            self._apply_lowpass_filter_cpu(cutoff)
    
    def reduce_noise_spectral_gpu(self, noise_factor=0.1):
        """GPU-accelerated spectral noise reduction using torch.fft"""
        # Convert to frequency domain
        fft_result = torch.fft.fft(self.audio_tensor)
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        
        # Estimate noise floor (bottom percentile)
        noise_floor = torch.quantile(magnitude, 0.1)
        
        # Reduce components below noise threshold
        magnitude_cleaned = torch.where(
            magnitude < noise_floor * (1 + noise_factor),
            magnitude * noise_factor,
            magnitude
        )
        
        # Reconstruct signal
        fft_cleaned = magnitude_cleaned * torch.exp(1j * phase)
        self.audio_tensor = torch.real(torch.fft.ifft(fft_cleaned))
        logger.debug("✓ Spectral noise reduction applied (GPU)")
    
    def normalize_audio_gpu(self):
        """Normalize audio using GPU operations"""
        max_val = torch.max(torch.abs(self.audio_tensor))
        if max_val > 0:
            self.audio_tensor = self.audio_tensor / max_val * 0.95
            logger.debug("✓ Audio normalized (GPU)")
    
    def boost_volume_gpu(self, gain_db=6):
        """Boost volume using GPU operations"""
        gain_linear = 10 ** (gain_db / 20)
        self.audio_tensor = self.audio_tensor * gain_linear
        
        # Prevent clipping
        max_val = torch.max(torch.abs(self.audio_tensor))
        if max_val > 1.0:
            self.audio_tensor = self.audio_tensor / max_val * 0.95
            logger.debug(f"✓ Volume boosted by {gain_db}dB (GPU, with clipping protection)")
        else:
            logger.debug(f"✓ Volume boosted by {gain_db}dB (GPU)")
    
    def apply_gentle_compression_gpu(self, threshold=0.7, ratio=3.0):
        """Apply gentle compression using GPU operations"""
        # Vectorized compression
        abs_audio = torch.abs(self.audio_tensor)
        over_threshold = abs_audio > threshold
        
        # Calculate compression for samples over threshold
        excess = abs_audio - threshold
        compressed_excess = excess / ratio
        compressed_magnitude = torch.where(
            over_threshold,
            threshold + compressed_excess,
            abs_audio
        )
        
        # Apply compressed magnitude while preserving sign
        self.audio_tensor = torch.sign(self.audio_tensor) * compressed_magnitude
        logger.debug(f"✓ Gentle compression applied (GPU, threshold: {threshold}, ratio: {ratio}:1)")
    
    def clean_audio_gpu(self, volume_boost_db=6, remove_crackles=True, 
                       apply_filters=True, reduce_noise=True):
        """Complete GPU-accelerated audio cleaning pipeline"""
        logger.info("Starting GPU-accelerated audio cleaning process...")
        
        # Basic cleaning
        self.remove_dc_offset()
        
        if remove_crackles:
            self.remove_clicks_pops_gpu()
        
        if apply_filters:
            self.apply_highpass_filter_gpu(cutoff=80)
            self.apply_lowpass_filter_gpu(cutoff=8000)
        
        if reduce_noise:
            self.reduce_noise_spectral_gpu()
        
        # Volume and dynamics
        self.normalize_audio_gpu()
        self.apply_gentle_compression_gpu()
        
        if volume_boost_db > 0:
            self.boost_volume_gpu(volume_boost_db)
        
        logger.info("GPU audio cleaning complete!")
        return self.audio_tensor.cpu().numpy()  # Return to CPU for final export
    
    # CPU fallback methods
    def _apply_highpass_filter_cpu(self, cutoff=80):
        """CPU fallback for high-pass filter"""
        audio_cpu = self.audio_tensor.cpu().numpy()
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff < 1.0:
            b, a = signal.butter(4, normalized_cutoff, btype='high')
            audio_cpu = signal.filtfilt(b, a, audio_cpu)
            self.audio_tensor = torch.from_numpy(audio_cpu).float().to(self.device)
            logger.debug(f"✓ High-pass filter applied (CPU fallback, cutoff: {cutoff} Hz)")
    
    def _apply_lowpass_filter_cpu(self, cutoff=8000):
        """CPU fallback for low-pass filter"""
        audio_cpu = self.audio_tensor.cpu().numpy()
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff < 1.0:
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            audio_cpu = signal.filtfilt(b, a, audio_cpu)
            self.audio_tensor = torch.from_numpy(audio_cpu).float().to(self.device)
            logger.debug(f"✓ Low-pass filter applied (CPU fallback, cutoff: {cutoff} Hz)")

# GPU-optimized audio processing functions
def clean_audio_from_file_gpu(file_path, volume_boost=6, remove_crackles=True, 
                             apply_filters=True, reduce_noise=True):
    """Clean audio from file path using GPU acceleration"""
    try:
        # Load audio using torchaudio for better GPU integration
        try:
            audio_tensor, sample_rate = torchaudio.load(file_path)
            audio_data = audio_tensor.mean(dim=0).numpy()  # Convert to mono
        except:
            # Fallback to librosa
            audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        # Clean audio using GPU
        cleaner = GPUAudioCleaner(audio_data, sample_rate, device)
        cleaned_audio = cleaner.clean_audio_gpu(
            volume_boost_db=volume_boost,
            remove_crackles=remove_crackles,
            apply_filters=apply_filters,
            reduce_noise=reduce_noise
        )
        
        return cleaned_audio, sample_rate
    except Exception as e:
        logger.error(f"Error cleaning audio: {e}")
        raise

def clean_audio_segment_gpu(audio_segment, volume_boost=6, remove_crackles=True, 
                           apply_filters=True, reduce_noise=True):
    """Clean AudioSegment object using GPU acceleration"""
    try:
        # Convert AudioSegment to numpy array
        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / (2**15)  # Normalize from int16 to float
        
        # If stereo, take first channel
        if audio_segment.channels == 2:
            audio_data = audio_data[::2]
        
        sample_rate = audio_segment.frame_rate
        
        # Clean audio using GPU
        cleaner = GPUAudioCleaner(audio_data, sample_rate, device)
        cleaned_audio = cleaner.clean_audio_gpu(
            volume_boost_db=volume_boost,
            remove_crackles=remove_crackles,
            apply_filters=apply_filters,
            reduce_noise=reduce_noise
        )
        
        # Convert back to AudioSegment
        cleaned_audio_int = (cleaned_audio * 32767).astype(np.int16)
        cleaned_segment = AudioSegment(
            cleaned_audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        return cleaned_segment
    except Exception as e:
        logger.error(f"Error cleaning audio segment: {e}")
        raise

# --- Model Loading with GPU Optimization ---
MODEL_LOADED = False
tts_model = None
try:
    MODEL_REPO = "kyutai/tts-1.6b-en_fr"
    VOICE_REPO = "kyutai/tts-voices"
    DEFAULT_VOICE = "Happy"
    
    logger.info(f"Loading model on device: {device}")
    checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
    
    # Load model with optimized settings for GPU
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, 
        n_q=32, 
        temp=0.6, 
        device=device
    )
    tts_model.voice_repo = VOICE_REPO
    
    # Enable optimizations if available
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            tts_model = torch.compile(tts_model, mode="max-autotune")
            logger.info("Model compiled with torch.compile for enhanced performance")
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
    
    MODEL_LOADED = True
    logger.info("Model loaded successfully with GPU optimizations.")
except Exception as e:
    MODEL_LOADED = False
    logger.exception(f"FATAL: Error loading model: {e}")
    def tts_model_placeholder():
        raise RuntimeError("TTS Model could not be loaded. The application cannot proceed.")
    tts_model = tts_model_placeholder

# --- Voice and SSML Configuration (unchanged) ---
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
    logger.info("Parsing input text...")
    segments = []
    stripped_text = ssml_text.strip()
    def escape_text_nodes(match):
        return match.group(1) + html.escape(match.group(2)) + match.group(3)
    try:
        sanitized_ssml = re.sub(r'(>)([^<>]*)(<)', escape_text_nodes, stripped_text)
        if not sanitized_ssml.startswith('<'):
             sanitized_ssml = html.escape(sanitized_ssml)
        root = ET.fromstring(sanitized_ssml)
    except ET.ParseError:
        logger.warning("SSML parsing failed even after sanitization. Treating input as plain text.")
        escaped_text = html.escape(stripped_text)
        ssml_for_text = f'<speak><voice name="{default_voice}">{escaped_text}</voice></speak>'
        root = ET.fromstring(ssml_for_text)
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

# --- Optimized Audio Generation ---
@torch.inference_mode()  # Disable gradient computation for inference
def generate_audio(ssml_text: str, default_voice: str, output_format: str = "mp3",
                  apply_cleaning: bool = False, volume_boost: float = 6.0, 
                  remove_crackles: bool = True, apply_filters: bool = True, 
                  reduce_noise: bool = True) -> str:
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded")

    logger.info("Generating audio from parsed segments...")
    segments = parse_ssml(ssml_text, default_voice)
    output_audio_segments = []

    # Process segments with GPU optimizations
    for idx, (voice, content) in enumerate(segments):
        logger.info(f"Processing segment {idx + 1}/{len(segments)}: Voice='{voice}'")
        if voice == "PAUSE":
            output_audio_segments.append(content)
            continue
        
        # Prepare script and attributes
        entries = tts_model.prepare_script([content], padding_between=1)
        voice_path = tts_model.get_voice_path(VOICE_OPTIONS.get(voice, VOICE_OPTIONS[default_voice]))
        attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.5)
        
        # Generate audio on GPU
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            result = tts_model.generate([entries], [attributes])

        # Decode audio efficiently
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[tts_model.delay_steps:]:
                decoded = tts_model.mimi.decode(frame[:, 1:, :])
                pcm = torch.clamp(decoded, -1, 1).cpu().numpy()[0, 0]
                pcms.append(pcm)
            pcm_data = np.concatenate(pcms, axis=-1)

        # Create temporary wav file
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

    logger.info("Normalizing final audio...")
    final_audio = normalize(final_audio)
    final_audio = final_audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    
    # Apply GPU-accelerated cleaning if requested
    if apply_cleaning:
        logger.info("Applying GPU-accelerated cleaning to final audio...")
        final_audio = clean_audio_segment_gpu(
            final_audio,
            volume_boost=volume_boost,
            remove_crackles=remove_crackles,
            apply_filters=apply_filters,
            reduce_noise=reduce_noise
        )
    
    # Export based on requested format
    if output_format.lower() == "wav":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_fp:
            final_audio.export(wav_fp.name, format="wav")
            logger.info(f"Final audio successfully exported to {wav_fp.name} as WAV")
            return wav_fp.name
    else:  # Default to MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_fp:
            final_audio.export(mp3_fp.name, format="mp3", bitrate="320k")
            logger.info(f"Final audio successfully exported to {mp3_fp.name} as MP3")
            return mp3_fp.name

# --- Pydantic Models (unchanged) ---
class TTSRequest(BaseModel):
    text: str
    voice_choice: str = DEFAULT_VOICE
    output_format: str = "mp3"
    apply_cleaning: bool = False
    volume_boost: float = 6.0
    remove_crackles: bool = True
    apply_filters: bool = True
    reduce_noise: bool = True

class AudioCleaningRequest(BaseModel):
    volume_boost: float = 6.0
    remove_crackles: bool = True
    apply_filters: bool = True
    reduce_noise: bool = True

# --- API Endpoints (updated for GPU optimization) ---
@app.post("/api/tts")
def tts_endpoint(request: TTSRequest):
    """Generate TTS audio with optional GPU-accelerated cleaning"""
    logger.info(f"TTS API endpoint hit. Format: {request.output_format}, Cleaning: {request.apply_cleaning}")
    
    if request.output_format.lower() not in ["mp3", "wav"]:
        return JSONResponse(
            status_code=400, 
            content={"detail": "Invalid output format. Use 'mp3' or 'wav'."}
        )
    
    try:
        output_path = generate_audio(
            request.text, 
            request.voice_choice, 
            output_format=request.output_format,
            apply_cleaning=request.apply_cleaning,
            volume_boost=request.volume_boost,
            remove_crackles=request.remove_crackles,
            apply_filters=request.apply_filters,
            reduce_noise=request.reduce_noise
        )
        
        if request.output_format.lower() == "wav":
            media_type = "audio/wav"
            filename = "generated_speech.wav"
        else:
            media_type = "audio/mpeg"
            filename = "generated_speech.mp3"
        
        return FileResponse(output_path, media_type=media_type, filename=filename)
        
    except ValueError as e:
        logger.error(f"Invalid input provided: {e}")
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        logger.exception("An internal error occurred during audio generation.")
        return JSONResponse(status_code=500, content={"detail": f"An internal error occurred: {str(e)}"})

@app.post("/api/clean-audio")
async def clean_audio_endpoint(
    audio_file: UploadFile = File(...),
    volume_boost: float = Form(6.0),
    remove_crackles: bool = Form(True),
    apply_filters: bool = Form(True),
    reduce_noise: bool = Form(True)
):
    """Clean uploaded audio file using GPU acceleration"""
    logger.info(f"GPU audio cleaning endpoint hit for file: {audio_file.filename}")
    
    if not audio_file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg')):
        return JSONResponse(
            status_code=400, 
            content={"detail": "Unsupported audio format. Please use MP3, WAV, FLAC, M4A, or OGG."}
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False) as temp_input:
            content = await audio_file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # Clean the audio using GPU acceleration
        cleaned_audio, sample_rate = clean_audio_from_file_gpu(
            temp_input_path,
            volume_boost=volume_boost,
            remove_crackles=remove_crackles,
            apply_filters=apply_filters,
            reduce_noise=reduce_noise
        )
        
        # Save cleaned audio as MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_output:
            cleaned_audio_int = (cleaned_audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                cleaned_audio_int.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            audio_segment.export(temp_output.name, format="mp3", bitrate="320k")
            temp_output_path = temp_output.name
        
        # Clean up input file
        os.unlink(temp_input_path)
        
        return FileResponse(
            temp_output_path, 
            media_type="audio/mpeg", 
            filename=f"cleaned_{audio_file.filename.rsplit('.', 1)[0]}.mp3"
        )
        
    except Exception as e:
        logger.exception("Error during GPU audio cleaning.")
        return JSONResponse(status_code=500, content={"detail": f"Audio cleaning failed: {str(e)}"})

@app.get("/api/voices")
def get_voices():
    """Get available voice options"""
    return {"voices": list(VOICE_OPTIONS.keys())}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
        "features": ["tts", "gpu_audio_cleaning", "ssml_support", "gpu_acceleration"]
    }

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "GPU-Optimized Kyutai TTS API with Hardware Acceleration",
        "version": "3.0.0",
        "device": str(device),
        "endpoints": {
            "tts": "/api/tts",
            "clean_audio": "/api/clean-audio", 
            "voices": "/api/voices",
            "health": "/api/health"
        },
        "features": [
            "GPU-accelerated Text-to-Speech with multiple voices and formats (MP3/WAV)",
            "SSML support",
            "GPU-accelerated audio cleaning (crackle removal, noise reduction)",
            "Hardware-optimized volume boosting and filtering",
            "Multiple audio format support for file cleaning",
            "Automatic mixed precision for improved performance"
        ]
    }

if __name__ == "__main__":
    if MODEL_LOADED:
        logger.info("Starting GPU-optimized server with hardware-accelerated audio processing.")
        logger.info(f"GPU Status: {torch.cuda.is_available()}, Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        logger.info("Launching FastAPI server on http://0.0.0.0:7861")
        uvicorn.run(app, host="0.0.0.0", port=7861)
    else:
        logger.error("Application will not start because the model failed to load.")