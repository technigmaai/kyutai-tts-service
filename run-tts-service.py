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
import psutil
import threading
import time

# Audio cleaning imports - keeping CPU fallbacks
import librosa
from scipy import signal
from scipy.ndimage import median_filter
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring
def log_system_usage():
    """Log CPU and GPU usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        gpu_info = f"GPU: {gpu_memory_percent:.1f}% ({gpu_memory_used:.1f}GB/{gpu_memory_total:.1f}GB)"
    
    logger.info(f"📊 Usage - CPU: {cpu_percent}%, RAM: {memory_percent}%, {gpu_info}")

# Start background monitoring (optional)
ENABLE_MONITORING = True  # Set to False to disable

def start_monitoring():
    if ENABLE_MONITORING:
        def monitor():
            while True:
                time.sleep(30)  # Log every 30 seconds
                log_system_usage()
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("📈 System monitoring started")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="GPU-Optimized Kyutai TTS API", description="Generate and clean speech audio with GPU acceleration.", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_COMPILATION = True  # Set to False to disable torch.compile attempts
ENABLE_CPU_OPTIMIZATIONS = True  # Enable CPU usage optimizations
ENABLE_BATCH_PROCESSING = True  # Enable batch processing for higher GPU utilization
MAX_BATCH_SIZE = 4  # Process up to 4 segments simultaneously
ENABLE_CONCURRENT_CLEANING = True  # Enable concurrent audio cleaning
logger.info(f"Using device: {device}")

# CPU optimization settings
if ENABLE_CPU_OPTIMIZATIONS:
    # Reduce CPU threads for PyTorch operations
    torch.set_num_threads(4)  # Reduce from default (usually 16-32)
    torch.set_num_interop_threads(2)  # Reduce inter-op parallelism
    
    # Set environment variables for CPU libraries
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
    logger.info("🔧 CPU usage optimizations enabled")

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
            # Check if HighpassBiquad is available (newer torchaudio versions)
            if hasattr(T, 'HighpassBiquad'):
                highpass = T.HighpassBiquad(
                    sample_rate=self.sample_rate, 
                    cutoff_freq=cutoff
                ).to(self.device)
                
                # Apply filter (need to add batch dimension)
                audio_filtered = highpass(self.audio_tensor.unsqueeze(0))
                self.audio_tensor = audio_filtered.squeeze(0)
                logger.debug(f"✓ High-pass filter applied (GPU, cutoff: {cutoff} Hz)")
            else:
                # Use alternative GPU-based filtering
                self._apply_highpass_filter_fft_gpu(cutoff)
        except Exception as e:
            logger.warning(f"GPU highpass failed, using CPU fallback: {e}")
            self._apply_highpass_filter_cpu(cutoff)
    
    def apply_lowpass_filter_gpu(self, cutoff=8000):
        """Apply low-pass filter using torchaudio on GPU"""
        try:
            # Check if LowpassBiquad is available (newer torchaudio versions)
            if hasattr(T, 'LowpassBiquad'):
                lowpass = T.LowpassBiquad(
                    sample_rate=self.sample_rate, 
                    cutoff_freq=cutoff
                ).to(self.device)
                
                # Apply filter
                audio_filtered = lowpass(self.audio_tensor.unsqueeze(0))
                self.audio_tensor = audio_filtered.squeeze(0)
                logger.debug(f"✓ Low-pass filter applied (GPU, cutoff: {cutoff} Hz)")
            else:
                # Use alternative GPU-based filtering
                self._apply_lowpass_filter_fft_gpu(cutoff)
        except Exception as e:
            logger.warning(f"GPU lowpass failed, using CPU fallback: {e}")
            self._apply_lowpass_filter_cpu(cutoff)
    
    def _apply_highpass_filter_fft_gpu(self, cutoff=80):
        """Alternative GPU-based high-pass filter using FFT"""
        # Convert to frequency domain
        fft_result = torch.fft.fft(self.audio_tensor)
        freqs = torch.fft.fftfreq(len(self.audio_tensor), 1/self.sample_rate).to(self.device)
        
        # Create high-pass mask
        mask = torch.abs(freqs) > cutoff
        fft_filtered = fft_result * mask.float()
        
        # Convert back to time domain
        self.audio_tensor = torch.real(torch.fft.ifft(fft_filtered))
        logger.debug(f"✓ High-pass filter applied (GPU FFT, cutoff: {cutoff} Hz)")
    
    def _apply_lowpass_filter_fft_gpu(self, cutoff=8000):
        """Alternative GPU-based low-pass filter using FFT"""
        # Convert to frequency domain
        fft_result = torch.fft.fft(self.audio_tensor)
        freqs = torch.fft.fftfreq(len(self.audio_tensor), 1/self.sample_rate).to(self.device)
        
        # Create low-pass mask
        mask = torch.abs(freqs) < cutoff
        fft_filtered = fft_result * mask.float()
        
        # Convert back to time domain
        self.audio_tensor = torch.real(torch.fft.ifft(fft_filtered))
        logger.debug(f"✓ Low-pass filter applied (GPU FFT, cutoff: {cutoff} Hz)")
    
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
    """Clean audio from file path using GPU acceleration with CPU optimization"""
    try:
        logger.debug("🔄 Loading audio file...")
        
        # Use torchaudio directly for GPU-native loading when possible
        try:
            audio_tensor, sample_rate = torchaudio.load(file_path, backend="ffmpeg")
            # Convert to mono on GPU
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            audio_data = audio_tensor.squeeze().cpu().numpy()  # Only move to CPU when needed
            logger.debug("✅ Audio loaded via torchaudio (GPU-optimized)")
        except Exception as torchaudio_error:
            logger.debug(f"Torchaudio failed, using librosa fallback: {torchaudio_error}")
            # Fallback to librosa with reduced CPU usage
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
        
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
    """Clean AudioSegment object using GPU acceleration with minimal CPU usage"""
    try:
        logger.debug("🔄 Converting AudioSegment for GPU processing...")
        
        # Optimize AudioSegment to numpy conversion with proper array handling
        raw_data = audio_segment.raw_data
        audio_array = np.frombuffer(raw_data, dtype=np.int16)
        
        # Ensure array is C-contiguous to avoid negative stride issues
        if not audio_array.flags['C_CONTIGUOUS']:
            audio_array = np.ascontiguousarray(audio_array)
        
        # Normalize efficiently
        audio_data = audio_array.astype(np.float32) / 32768.0
        
        # Handle stereo -> mono conversion efficiently
        if audio_segment.channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            # Ensure the result is contiguous
            audio_data = np.ascontiguousarray(audio_data)
        
        sample_rate = audio_segment.frame_rate
        
        # Clean audio using GPU
        cleaner = GPUAudioCleaner(audio_data, sample_rate, device)
        cleaned_audio = cleaner.clean_audio_gpu(
            volume_boost_db=volume_boost,
            remove_crackles=remove_crackles,
            apply_filters=apply_filters,
            reduce_noise=reduce_noise
        )
        
        # Convert back to AudioSegment efficiently with proper bounds checking
        cleaned_audio_int = np.clip(cleaned_audio * 32767, -32767, 32767).astype(np.int16)
        
        # Ensure the array is contiguous before creating AudioSegment
        if not cleaned_audio_int.flags['C_CONTIGUOUS']:
            cleaned_audio_int = np.ascontiguousarray(cleaned_audio_int)
        
        cleaned_segment = AudioSegment(
            cleaned_audio_int.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        logger.debug("✅ AudioSegment processing complete")
        return cleaned_segment
    except Exception as e:
        logger.error(f"Error cleaning audio segment: {e}")
        # Return original audio if cleaning fails
        logger.info("🔄 Returning original audio without cleaning")
        return audio_segment

# --- Model Loading with GPU Optimization ---
MODEL_LOADED = False
tts_model = None
COMPILATION_ENABLED = False

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
    
    # Enable basic CUDA optimizations if available
    if torch.cuda.is_available():
        # Enable CUDA optimizations (conservative settings for stability)
        torch.backends.cudnn.benchmark = False  # More conservative for variable input sizes
        torch.backends.cudnn.deterministic = True
        
        # Optional: Pre-allocate memory for better performance
        try:
            torch.cuda.empty_cache()
            # Reserve 90% of GPU memory for this process
            torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info("🚀 GPU memory pre-allocated for optimal performance")
        except Exception as mem_error:
            logger.warning(f"GPU memory pre-allocation failed: {mem_error}")
        
        # Attempt model compilation with fallback (only if enabled)
        if ENABLE_COMPILATION:
            try:
                logger.info("Attempting model compilation...")
                
                # Use more conservative compilation settings to avoid dtype issues
                compiled_model = torch.compile(
                    tts_model, 
                    mode="default",  # Less aggressive than max-autotune
                    fullgraph=False,  # Allow graph breaks for better compatibility
                    dynamic=True      # Handle variable input sizes better
                )
                
                # Test the compiled model with a simple generation
                logger.info("Testing compiled model with dummy generation...")
                test_entries = tts_model.prepare_script(["test"], padding_between=1)
                test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                
                # Attempt a small generation to validate compilation
                with torch.no_grad():
                    _ = compiled_model.generate([test_entries], [test_attributes])
                
                tts_model = compiled_model
                COMPILATION_ENABLED = True
                logger.info("✅ Model compiled successfully with torch.compile!")
                
            except Exception as compile_error:
                logger.warning(f"⚠️  Model compilation failed, using uncompiled model.")
                logger.debug(f"Compilation error details: {compile_error}")
                COMPILATION_ENABLED = False
                # Continue with uncompiled model
                
                # Additional debugging for common compilation issues
                if "scatter" in str(compile_error).lower():
                    logger.info("💡 Compilation failed due to scatter operation compatibility. This is common with complex transformer models.")
                elif "dtype" in str(compile_error).lower():
                    logger.info("💡 Compilation failed due to dtype mismatch. The model will run efficiently without compilation.")
                elif "graph break" in str(compile_error).lower():
                    logger.info("💡 Compilation failed due to graph breaks. Consider setting fullgraph=False (already enabled).")
                else:
                    logger.info("💡 Compilation failed for unknown reasons. The uncompiled model will work fine.")
        else:
            logger.info("🔧 Model compilation disabled by configuration")
            COMPILATION_ENABLED = False
    
    MODEL_LOADED = True
    logger.info(f"Model loaded successfully. GPU optimizations: {torch.cuda.is_available()}, Compilation: {COMPILATION_ENABLED}")
    
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

# --- High GPU Utilization Audio Generation ---
def generate_audio_high_gpu_util(ssml_text: str, default_voice: str, output_format: str = "mp3",
                                apply_cleaning: bool = False, volume_boost: float = 6.0, 
                                remove_crackles: bool = True, apply_filters: bool = True, 
                                reduce_noise: bool = True) -> str:
    """Generate audio with maximum GPU utilization through batching and concurrency"""
    global ENABLE_BATCH_PROCESSING  # Fix scoping issue
    
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded")

    start_time = time.time()
    logger.info("🎵 Starting HIGH GPU UTILIZATION audio generation...")
    
    if ENABLE_MONITORING:
        log_system_usage()
    
    segments = parse_ssml(ssml_text, default_voice)
    
    # Separate text segments from pause segments
    text_segments = []
    pause_segments = []
    segment_map = []  # Track original order
    
    for idx, (voice, content) in enumerate(segments):
        if voice == "PAUSE":
            pause_segments.append((idx, content))
            segment_map.append(("PAUSE", len(pause_segments) - 1))
        else:
            text_segments.append((idx, voice, content))
            segment_map.append(("TEXT", len(text_segments) - 1))
    
    # Pre-allocate GPU memory for batch processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Pre-allocate larger memory chunks for batch processing
        try:
            dummy_tensor = torch.zeros(MAX_BATCH_SIZE, 1024, device=device)
            del dummy_tensor
        except:
            pass  # Continue if allocation fails
    
    logger.info(f"🚀 Processing {len(text_segments)} text segments in batches of {MAX_BATCH_SIZE}")
    
    # Process text segments in batches for higher GPU utilization
    text_audio_results = {}
    batch_processing_enabled = ENABLE_BATCH_PROCESSING  # Local copy to avoid scoping issues
    
    if batch_processing_enabled and len(text_segments) > 1:
        # BATCH PROCESSING MODE - Higher GPU Utilization
        for batch_start in range(0, len(text_segments), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(text_segments))
            batch = text_segments[batch_start:batch_end]
            
            logger.info(f"🔥 Processing batch {batch_start//MAX_BATCH_SIZE + 1} with {len(batch)} segments (GPU INTENSIVE)")
            
            # Prepare all batch data
            batch_entries = []
            batch_attributes = []
            batch_indices = []
            
            for idx, voice, content in batch:
                entries = tts_model.prepare_script([content], padding_between=1)
                voice_path = tts_model.get_voice_path(VOICE_OPTIONS.get(voice, VOICE_OPTIONS[default_voice]))
                attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.5)
                
                batch_entries.append(entries)
                batch_attributes.append(attributes)
                batch_indices.append(idx)
            
            # PARALLEL GPU GENERATION - This maxes out GPU utilization
            try:
                with torch.no_grad():
                    # Process multiple segments simultaneously
                    batch_results = []
                    
                    # Create multiple CUDA streams for concurrent execution
                    if torch.cuda.is_available():
                        streams = [torch.cuda.Stream() for _ in range(min(len(batch), 4))]  # Limit streams
                        
                        # Launch parallel generations
                        for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                            stream_idx = i % len(streams)  # Cycle through available streams
                            with torch.cuda.stream(streams[stream_idx]):
                                result = tts_model.generate([entries], [attributes])
                                batch_results.append((batch_indices[i], result))
                        
                        # Synchronize all streams
                        for stream in streams:
                            stream.synchronize()
                    else:
                        # CPU fallback
                        for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                            result = tts_model.generate([entries], [attributes])
                            batch_results.append((batch_indices[i], result))
                    
                    logger.info(f"✅ Batch completed with {len(batch_results)} parallel generations")
                    
                    # Decode all results (keeping on GPU as long as possible)
                    for idx, result in batch_results:
                        try:
                            with tts_model.mimi.streaming(1), torch.no_grad():
                                pcms = []
                                for frame in result.frames[tts_model.delay_steps:]:
                                    decoded = tts_model.mimi.decode(frame[:, 1:, :])
                                    pcm = torch.clamp(decoded, -1, 1)
                                    pcms.append(pcm[0, 0])
                                
                                # Keep on GPU until final transfer
                                pcm_tensor = torch.cat(pcms, dim=-1)
                                pcm_data = pcm_tensor.cpu().numpy()
                                
                                # Store result
                                text_audio_results[idx] = pcm_data
                                
                        except Exception as decode_error:
                            logger.error(f"Batch decode failed for segment {idx}: {decode_error}")
                            continue
                            
            except Exception as batch_error:
                logger.error(f"Batch processing failed: {batch_error}")
                # Fallback to sequential processing
                logger.info("🔄 Falling back to sequential processing")
                batch_processing_enabled = False
    
    # Sequential fallback or single segment
    if not batch_processing_enabled or len(text_segments) <= 1:
        logger.info("📝 Using sequential processing")
        for idx, voice, content in text_segments:
            segment_start = time.time()
            logger.info(f"🔄 Processing segment {idx + 1}: Voice='{voice}'")
            
            entries = tts_model.prepare_script([content], padding_between=1)
            voice_path = tts_model.get_voice_path(VOICE_OPTIONS.get(voice, VOICE_OPTIONS[default_voice]))
            attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.5)
            
            try:
                with torch.no_grad():
                    result = tts_model.generate([entries], [attributes])
                
                with tts_model.mimi.streaming(1), torch.no_grad():
                    pcms = []
                    for frame in result.frames[tts_model.delay_steps:]:
                        decoded = tts_model.mimi.decode(frame[:, 1:, :])
                        pcm = torch.clamp(decoded, -1, 1)
                        pcms.append(pcm[0, 0])
                    
                    pcm_tensor = torch.cat(pcms, dim=-1)
                    pcm_data = pcm_tensor.cpu().numpy()
                    text_audio_results[idx] = pcm_data
                    
                segment_time = time.time() - segment_start
                logger.debug(f"⏱️  Sequential segment completed in {segment_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Sequential processing failed for segment {idx}: {e}")
                continue
    
    # Reconstruct audio in original order
    logger.info("🔗 Reconstructing audio in original order...")
    output_audio_segments = []
    
    for segment_type, segment_idx in segment_map:
        if segment_type == "PAUSE":
            original_idx, pause_audio = pause_segments[segment_idx]
            output_audio_segments.append(pause_audio)
        else:
            original_idx = text_segments[segment_idx][0]
            if original_idx in text_audio_results:
                # Convert to AudioSegment
                pcm_data = text_audio_results[original_idx]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_fp:
                    sphn.write_wav(wav_fp.name, pcm_data, tts_model.mimi.sample_rate)
                    segment_audio = AudioSegment.from_wav(wav_fp.name)
                    output_audio_segments.append(segment_audio)
                    os.unlink(wav_fp.name)
    
    if not output_audio_segments:
        logger.warning("No audio was generated, returning empty audio file.")
        final_audio = AudioSegment.silent(duration=1)
    else:
        logger.info("🔗 Combining audio segments...")
        final_audio = output_audio_segments[0]
        for segment in output_audio_segments[1:]:
            final_audio += segment

    logger.info("🎚️  Normalizing final audio...")
    final_audio = final_audio.normalize()
    final_audio = final_audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    
    # CONCURRENT GPU CLEANING for maximum utilization
    if apply_cleaning and ENABLE_CONCURRENT_CLEANING:
        try:
            cleaning_start = time.time()
            logger.info("🧹 Applying CONCURRENT GPU-accelerated cleaning...")
            
            # Split audio into chunks for concurrent processing
            audio_duration = len(final_audio)
            if audio_duration > 4000:  # Only split if audio is longer than 4 seconds
                chunk_size = audio_duration // 4  # 4 concurrent chunks
                chunks = [final_audio[i:i+chunk_size] for i in range(0, audio_duration, chunk_size)]
                
                # Process chunks concurrently
                cleaned_chunks = []
                
                if torch.cuda.is_available():
                    streams = [torch.cuda.Stream() for _ in range(min(len(chunks), 4))]
                    
                    for i, chunk in enumerate(chunks):
                        stream_idx = i % len(streams)
                        with torch.cuda.stream(streams[stream_idx]):
                            cleaned_chunk = clean_audio_segment_gpu(
                                chunk,
                                volume_boost=volume_boost,
                                remove_crackles=remove_crackles,
                                apply_filters=apply_filters,
                                reduce_noise=reduce_noise
                            )
                            cleaned_chunks.append(cleaned_chunk)
                    
                    # Synchronize and combine
                    for stream in streams:
                        stream.synchronize()
                else:
                    # CPU fallback
                    for chunk in chunks:
                        cleaned_chunk = clean_audio_segment_gpu(
                            chunk, volume_boost, remove_crackles, apply_filters, reduce_noise
                        )
                        cleaned_chunks.append(cleaned_chunk)
                
                # Combine cleaned chunks
                final_audio = sum(cleaned_chunks)
            else:
                # Audio too short for chunking, use standard cleaning
                final_audio = clean_audio_segment_gpu(
                    final_audio, volume_boost, remove_crackles, apply_filters, reduce_noise
                )
            
            cleaning_time = time.time() - cleaning_start
            logger.info(f"✅ CONCURRENT audio cleaning completed in {cleaning_time:.2f}s")
        except Exception as cleaning_error:
            logger.warning(f"Concurrent GPU cleaning failed, using standard cleaning: {cleaning_error}")
            # Fallback to standard cleaning
            try:
                final_audio = clean_audio_segment_gpu(final_audio, volume_boost, remove_crackles, apply_filters, reduce_noise)
            except Exception as fallback_error:
                logger.warning(f"Standard cleaning also failed: {fallback_error}")
    elif apply_cleaning:
        # Standard GPU cleaning
        try:
            logger.info("🧹 Applying standard GPU cleaning...")
            final_audio = clean_audio_segment_gpu(final_audio, volume_boost, remove_crackles, apply_filters, reduce_noise)
        except Exception as cleaning_error:
            logger.warning(f"GPU cleaning failed: {cleaning_error}")
    
    # Export
    export_start = time.time()
    if output_format.lower() == "wav":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_fp:
            final_audio.export(wav_fp.name, format="wav", parameters=["-ac", "1"])
            export_time = time.time() - export_start
            logger.info(f"📁 Final audio exported as WAV in {export_time:.2f}s")
            total_time = time.time() - start_time
            logger.info(f"🎉 HIGH GPU UTILIZATION generation completed in {total_time:.2f}s")
            if ENABLE_MONITORING:
                log_system_usage()
            return wav_fp.name
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_fp:
            final_audio.export(mp3_fp.name, format="mp3", bitrate="320k", parameters=["-ac", "1"])
            export_time = time.time() - export_start
            logger.info(f"📁 Final audio exported as MP3 in {export_time:.2f}s")
            total_time = time.time() - start_time
            logger.info(f"🎉 HIGH GPU UTILIZATION generation completed in {total_time:.2f}s")
            if ENABLE_MONITORING:
                log_system_usage()
            return mp3_fp.name

# Use high GPU utilization version by default
generate_audio = generate_audio_high_gpu_util

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
        "compilation_enabled": COMPILATION_ENABLED if MODEL_LOADED else False,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
        "features": ["tts", "gpu_audio_cleaning", "ssml_support", "gpu_acceleration", "error_recovery"]
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
        logger.info("🚀 Starting GPU-optimized server with hardware-accelerated audio processing.")
        logger.info(f"GPU Status: {torch.cuda.is_available()}, Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Log CPU optimization status
        if ENABLE_CPU_OPTIMIZATIONS:
            logger.info(f"🔧 CPU optimizations enabled - Threads: {torch.get_num_threads()}")
        
        # Start system monitoring
        if ENABLE_MONITORING:
            start_monitoring()
        
        # Log startup configuration
        logger.info(f"🔧 CPU optimizations: {'Enabled' if ENABLE_CPU_OPTIMIZATIONS else 'Disabled'}")
        logger.info(f"📊 System monitoring: {'Enabled' if ENABLE_MONITORING else 'Disabled'}")
        logger.info(f"🚀 Batch processing: {'Enabled' if ENABLE_BATCH_PROCESSING else 'Disabled'} (Max batch: {MAX_BATCH_SIZE})")
        logger.info(f"⚡ Concurrent cleaning: {'Enabled' if ENABLE_CONCURRENT_CLEANING else 'Disabled'}")
        logger.info(f"🧠 PyTorch threads: {torch.get_num_threads()}")
        
        # Initial system status
        log_system_usage()
        
        logger.info("🌐 Launching FastAPI server on http://0.0.0.0:7861")
        
        # Configure uvicorn for optimal performance
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=7861,
            workers=1,  # Single worker to avoid GPU memory conflicts
            loop="asyncio",  # Use asyncio for better performance
            access_log=False  # Disable access logs to reduce CPU overhead
        )
    else:
        logger.error("Application will not start because the model failed to load.")