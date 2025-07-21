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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Audio cleaning imports - keeping CPU fallbacks
import librosa
from scipy import signal
from scipy.ndimage import median_filter
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# Setup logging first (before memory pool)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Sophisticated GPU Memory Pooling System ---
class GPUMemoryPool:
    """Advanced GPU memory pooling system for efficient memory management"""
    
    def __init__(self, device=None, max_memory_fraction=0.85):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_fraction = max_memory_fraction
        self.pools = {}  # Size -> list of tensors
        self.allocated_tensors = {}  # id(tensor) -> size
        self.total_allocated = 0
        self.max_allocated = 0
        self.cleanup_threshold = 0.7  # Cleanup when 70% of max memory is used
        self.pool_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        
        # Adaptive memory management
        self.usage_history = []  # Track memory usage patterns
        self.adaptive_cleanup = True
        self.peak_usage_threshold = 0.8  # Trigger cleanup at 80% peak usage
        
        if torch.cuda.is_available():
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.max_allocated_memory = int(self.total_gpu_memory * max_memory_fraction)
            logger.info(f"🧠 GPU Memory Pool initialized - Max: {self.max_allocated_memory / 1024**3:.2f}GB")
        else:
            self.total_gpu_memory = 0
            self.max_allocated_memory = 0
            logger.info("🧠 GPU Memory Pool initialized (CPU mode)")
    
    def update_usage_history(self):
        """Track memory usage patterns for adaptive management"""
        current_usage = self.total_allocated / self.max_allocated_memory
        self.usage_history.append(current_usage)
        
        # Keep last 100 measurements
        if len(self.usage_history) > 100:
            self.usage_history.pop(0)
        
        # Adaptive cleanup threshold based on usage patterns
        if len(self.usage_history) > 10:
            avg_usage = sum(self.usage_history) / len(self.usage_history)
            peak_usage = max(self.usage_history)
            
            # Adjust cleanup threshold based on usage patterns
            if peak_usage > 0.9:  # High peak usage
                self.cleanup_threshold = min(0.6, self.cleanup_threshold * 0.95)
            elif avg_usage < 0.3:  # Low average usage
                self.cleanup_threshold = min(0.8, self.cleanup_threshold * 1.05)
    
    def get_nearest_pool_size(self, size):
        """Find the nearest pool size for efficient allocation"""
        for pool_size in self.pool_sizes:
            if pool_size >= size:
                return pool_size
        return size  # If larger than any pool, allocate exact size
    
    def allocate(self, size, dtype=torch.float32, device=None):
        """Allocate tensor from pool or create new one"""
        if device is None:
            device = self.device
            
        if not torch.cuda.is_available() or device.type == 'cpu':
            # CPU fallback
            return torch.empty(size, dtype=dtype, device=device)
        
        # Update usage history for adaptive management
        self.update_usage_history()
        
        # Check if we need cleanup
        if self.total_allocated > self.max_allocated_memory * self.cleanup_threshold:
            self.cleanup()
        
        # Find appropriate pool size
        pool_size = self.get_nearest_pool_size(size)
        
        # Try to get from pool
        if pool_size in self.pools and self.pools[pool_size]:
            tensor = self.pools[pool_size].pop()
            # Resize if needed
            if tensor.numel() >= size:
                tensor = tensor[:size] if len(tensor.shape) == 1 else tensor[:size, ...]
            else:
                # Pool tensor too small, create new one
                tensor = torch.empty(size, dtype=dtype, device=device)
        else:
            # Create new tensor
            tensor = torch.empty(size, dtype=dtype, device=device)
        
        # Track allocation
        tensor_id = id(tensor)
        self.allocated_tensors[tensor_id] = size
        self.total_allocated += size * tensor.element_size()
        self.max_allocated = max(self.max_allocated, self.total_allocated)
        
        return tensor
    
    def free(self, tensor):
        """Return tensor to pool for reuse"""
        if not torch.cuda.is_available() or tensor.device.type == 'cpu':
            return
        
        tensor_id = id(tensor)
        if tensor_id in self.allocated_tensors:
            size = self.allocated_tensors.pop(tensor_id)
            self.total_allocated -= size * tensor.element_size()
            
            # Clear tensor data and add to pool
            tensor.zero_()
            pool_size = self.get_nearest_pool_size(size)
            
            if pool_size not in self.pools:
                self.pools[pool_size] = []
            
            # Limit pool size to prevent memory bloat
            if len(self.pools[pool_size]) < 10:
                self.pools[pool_size].append(tensor)
    
    def cleanup(self, force=False):
        """Clean up pools and free memory"""
        if not torch.cuda.is_available():
            return
        
        # Clear all pools
        for pool_size in self.pools:
            self.pools[pool_size].clear()
        
        # Force garbage collection
        if force:
            torch.cuda.empty_cache()
        
        logger.debug(f"🧹 GPU Memory Pool cleaned - Allocated: {self.total_allocated / 1024**3:.2f}GB")
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return {
                "device": "cpu",
                "total_allocated": 0,
                "max_allocated": 0,
                "pool_count": 0
            }
        
        pool_count = sum(len(tensors) for tensors in self.pools.values())
        return {
            "device": str(self.device),
            "total_allocated_gb": self.total_allocated / 1024**3,
            "max_allocated_gb": self.max_allocated / 1024**3,
            "max_memory_gb": self.max_allocated_memory / 1024**3,
            "utilization_percent": (self.total_allocated / self.max_allocated_memory) * 100,
            "pool_count": pool_count,
            "active_tensors": len(self.allocated_tensors)
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(force=True)

# Global memory pool instance
gpu_memory_pool = GPUMemoryPool() if torch.cuda.is_available() else None

# Memory pool context manager
class MemoryPoolContext:
    """Context manager for GPU memory pool operations"""
    
    def __init__(self, pool=None):
        self.pool = pool or gpu_memory_pool
    
    def __enter__(self):
        return self.pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.cleanup()

# Enhanced memory allocation functions
def allocate_gpu_tensor(size, dtype=torch.float32, device=None):
    """Allocate tensor using memory pool"""
    if gpu_memory_pool:
        return gpu_memory_pool.allocate(size, dtype, device)
    else:
        device = device or torch.device("cpu")
        return torch.empty(size, dtype=dtype, device=device)

def free_gpu_tensor(tensor):
    """Free tensor back to memory pool"""
    if gpu_memory_pool and tensor.device.type == 'cuda':
        gpu_memory_pool.free(tensor)

# Performance monitoring
def log_system_usage():
    """Log CPU and GPU usage with memory pool statistics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        gpu_info = f"GPU: {gpu_memory_percent:.1f}% ({gpu_memory_used:.1f}GB/{gpu_memory_total:.1f}GB)"
        
        # Add memory pool statistics
        if gpu_memory_pool:
            pool_stats = gpu_memory_pool.get_memory_stats()
            pool_info = f" | Pool: {pool_stats['total_allocated_gb']:.2f}GB/{pool_stats['max_memory_gb']:.2f}GB ({pool_stats['utilization_percent']:.1f}%) | Tensors: {pool_stats['active_tensors']}"
            gpu_info += pool_info
    
    logger.info(f"📊 Usage - CPU: {cpu_percent}%, RAM: {memory_percent}%, {gpu_info}")

# Start background monitoring (optional)
ENABLE_MONITORING = True  # Set to False to disable

def start_monitoring():
    if ENABLE_MONITORING:
        def monitor():
            cleanup_counter = 0
            while True:
                time.sleep(30)  # Log every 30 seconds
                log_system_usage()
                
                # Periodic memory pool cleanup (every 10 minutes)
                cleanup_counter += 1
                if cleanup_counter >= 20 and gpu_memory_pool:  # 20 * 30s = 10 minutes
                    gpu_memory_pool.cleanup()
                    cleanup_counter = 0
                    logger.info("🧹 Periodic GPU memory pool cleanup completed")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("📈 System monitoring started with periodic memory cleanup")

# FastAPI setup
app = FastAPI(title="GPU-Optimized Kyutai TTS API", description="Generate and clean speech audio with GPU acceleration.", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_COMPILATION = False  # Temporarily disable compilation to focus on other optimizations
ENABLE_CPU_OPTIMIZATIONS = True  # Enable CPU usage optimizations
ENABLE_BATCH_PROCESSING = True  # Enable batch processing for higher GPU utilization
MAX_BATCH_SIZE = 16  # Process up to 16 segments simultaneously (increased from 8)
ENABLE_CONCURRENT_CLEANING = True  # Enable concurrent audio cleaning
ENABLE_FAST_MODE = True  # Enable fast mode without audio cleaning for speed
ENABLE_MODEL_QUANTIZATION = True  # Enable model quantization for speed
ENABLE_PARALLEL_PROCESSING = True  # Enable parallel processing
MAX_CONCURRENT_REQUESTS = 4  # Allow more concurrent requests

# Request queue management
from queue import Queue
from threading import Lock
import asyncio
import hashlib
import json

class AudioCache:
    """Cache for frequently requested audio generations"""
    
    def __init__(self, max_cache_size=100, cache_dir="/tmp/tts_cache"):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.cache_dir = cache_dir
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def generate_cache_key(self, text, voice, output_format, **kwargs):
        """Generate a unique cache key for the request"""
        # Create a hash of the request parameters
        cache_data = {
            "text": text,
            "voice": voice,
            "output_format": output_format,
            **kwargs
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_audio(self, cache_key):
        """Retrieve cached audio if available"""
        if cache_key in self.cache:
            file_path = self.cache[cache_key]["file_path"]
            if os.path.exists(file_path):
                self.cache_stats["hits"] += 1
                logger.info(f"🎯 Cache hit for key: {cache_key[:8]}...")
                return file_path
        
        self.cache_stats["misses"] += 1
        return None
    
    def cache_audio(self, cache_key, file_path, metadata=None):
        """Cache the generated audio"""
        if len(self.cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self.cache))
            old_file = self.cache[oldest_key]["file_path"]
            if os.path.exists(old_file):
                os.unlink(old_file)
            del self.cache[oldest_key]
            self.cache_stats["evictions"] += 1
        
        self.cache[cache_key] = {
            "file_path": file_path,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        logger.info(f"💾 Cached audio for key: {cache_key[:8]}...")
    
    def get_stats(self):
        """Get cache statistics"""
        hit_rate = self.cache_stats["hits"] / max(1, self.cache_stats["hits"] + self.cache_stats["misses"]) * 100
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "hit_rate_percent": hit_rate,
            **self.cache_stats
        }

# Global cache instance
audio_cache = AudioCache()

class RequestQueue:
    """Manage concurrent requests to prevent memory overload"""
    
    def __init__(self, max_concurrent=3, max_queue_size=10):
        self.queue = Queue(maxsize=max_queue_size)
        self.active_requests = 0
        self.max_concurrent = max_concurrent
        self.lock = Lock()
        self.request_stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0
        }
    
    async def submit_request(self, request_func, *args, **kwargs):
        """Submit a request to the queue"""
        if self.active_requests >= self.max_concurrent:
            # Queue is full, reject request
            raise RuntimeError("Server is at maximum capacity. Please try again later.")
        
        with self.lock:
            self.active_requests += 1
            self.request_stats["total_requests"] += 1
        
        try:
            start_time = time.time()
            result = await request_func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Update statistics
            with self.lock:
                self.request_stats["completed_requests"] += 1
                # Update average processing time
                total_completed = self.request_stats["completed_requests"]
                current_avg = self.request_stats["avg_processing_time"]
                self.request_stats["avg_processing_time"] = (current_avg * (total_completed - 1) + processing_time) / total_completed
            
            return result
        except Exception as e:
            with self.lock:
                self.request_stats["failed_requests"] += 1
            raise e
        finally:
            with self.lock:
                self.active_requests -= 1
    
    def get_stats(self):
        """Get queue statistics"""
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "max_concurrent": self.max_concurrent,
                "queue_size": self.queue.qsize(),
                "queue_max_size": self.queue.maxsize,
                **self.request_stats
            }

# Global request queue
request_queue = RequestQueue()

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
        
        # Convert to torch tensor and move to GPU using memory pool
        if isinstance(audio_data, np.ndarray):
            # Use memory pool for allocation
            if gpu_memory_pool and self.device.type == 'cuda':
                size = audio_data.size
                self.audio_tensor = allocate_gpu_tensor(size, torch.float32, self.device)
                self.audio_tensor.copy_(torch.from_numpy(audio_data).float())
            else:
                self.audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
        else:
            if gpu_memory_pool and self.device.type == 'cuda':
                size = audio_data.numel()
                self.audio_tensor = allocate_gpu_tensor(size, audio_data.dtype, self.device)
                self.audio_tensor.copy_(audio_data.float())
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
        result = self.audio_tensor.cpu().numpy()  # Return to CPU for final export
        
        # Clean up GPU memory
        if gpu_memory_pool:
            free_gpu_tensor(self.audio_tensor)
        
        return result
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'audio_tensor') and gpu_memory_pool:
            free_gpu_tensor(self.audio_tensor)
    
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
    
    # Apply model quantization for faster inference if enabled
    if ENABLE_MODEL_QUANTIZATION and torch.cuda.is_available():
        try:
            logger.info("🚀 Applying model quantization for faster inference...")
            # Note: TTSModel doesn't support .half() method, so we skip quantization
            # The model will use default precision which is still optimized
            logger.info("ℹ️ Model quantization skipped - TTSModel doesn't support .half()")
        except Exception as quant_error:
            logger.warning(f"⚠️ Model quantization failed: {quant_error}")
    
    # Enable basic CUDA optimizations if available
    if torch.cuda.is_available():
        # Enable CUDA optimizations (more aggressive settings for speed)
        torch.backends.cudnn.benchmark = True  # Enable for better performance
        torch.backends.cudnn.deterministic = False  # Disable for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for speed
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN
        
        # Additional optimizations for better performance
        torch.backends.cuda.enable_flash_sdp(True)  # Enable flash attention if available
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Enable memory efficient attention
        torch.backends.cuda.enable_math_sdp(True)  # Enable math attention
        
        # Initialize memory pool and pre-allocate memory for better performance
        try:
            torch.cuda.empty_cache()
            # Reserve 90% of GPU memory for this process (more aggressive)
            torch.cuda.set_per_process_memory_fraction(0.90)
            
            # Pre-allocate some common tensor sizes in the memory pool
            if gpu_memory_pool:
                common_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
                for size in common_sizes:
                    for _ in range(5):  # Pre-allocate 5 tensors of each size
                        dummy_tensor = allocate_gpu_tensor(size, torch.float32, device)
                        free_gpu_tensor(dummy_tensor)
                
                logger.info("🚀 GPU memory pool pre-allocated for optimal performance")
            else:
                logger.info("🚀 GPU memory pre-allocated for optimal performance")
        except Exception as mem_error:
            logger.warning(f"GPU memory pre-allocation failed: {mem_error}")
        
        # Enhanced model compilation with multiple fallback strategies
        if ENABLE_COMPILATION:
            logger.info("🚀 Attempting enhanced model compilation with multiple strategies...")
            
            # Strategy 1: Simple compilation without backend specification
            try:
                logger.info("📈 Strategy 1: Simple compilation without backend...")
                compiled_model = torch.compile(
                    tts_model, 
                    mode="reduce-overhead",  # More aggressive than "default"
                    fullgraph=False,  # Allow graph breaks for better compatibility
                    dynamic=True      # Handle variable input sizes better
                )
                
                # Test the compiled model with a simple generation
                logger.info("🧪 Testing compiled model with dummy generation...")
                test_entries = tts_model.prepare_script(["test"], padding_between=1)
                test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                
                # Attempt a small generation to validate compilation
                with torch.no_grad():
                    _ = compiled_model.generate([test_entries], [test_attributes])
                
                tts_model = compiled_model
                COMPILATION_ENABLED = True
                logger.info("✅ Strategy 1 SUCCESS! Model compiled with reduce-overhead mode!")
                
            except Exception as compile_error_1:
                logger.warning(f"⚠️  Strategy 1 failed: {str(compile_error_1)}")
                logger.debug(f"Strategy 1 error details: {type(compile_error_1).__name__}: {compile_error_1}")
                
                # Strategy 2: Conservative compilation with default mode
                try:
                    logger.info("📈 Strategy 2: Conservative compilation with default mode...")
                    compiled_model = torch.compile(
                        tts_model, 
                        mode="default",  # Conservative mode
                        fullgraph=False,  # Allow graph breaks
                        dynamic=True      # Handle variable inputs
                    )
                    
                    # Test the compiled model
                    logger.info("🧪 Testing compiled model...")
                    test_entries = tts_model.prepare_script(["test"], padding_between=1)
                    test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                    test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                    
                    with torch.no_grad():
                        _ = compiled_model.generate([test_entries], [test_attributes])
                    
                    tts_model = compiled_model
                    COMPILATION_ENABLED = True
                    logger.info("✅ Strategy 2 SUCCESS! Model compiled with default mode!")
                    
                except Exception as compile_error_2:
                    logger.warning(f"⚠️  Strategy 2 failed: {str(compile_error_2)}")
                    logger.debug(f"Strategy 2 error details: {type(compile_error_2).__name__}: {compile_error_2}")
                    
                    # Strategy 3: Minimal compilation with max-autotune-no-cudagraphs
                    try:
                        logger.info("📈 Strategy 3: Minimal compilation with max-autotune-no-cudagraphs...")
                        compiled_model = torch.compile(
                            tts_model, 
                            mode="max-autotune-no-cudagraphs",  # Minimal compilation
                            fullgraph=False,  # Allow graph breaks
                            dynamic=True      # Handle variable inputs
                        )
                        
                        # Test the compiled model
                        logger.info("🧪 Testing compiled model...")
                        test_entries = tts_model.prepare_script(["test"], padding_between=1)
                        test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                        test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                        
                        with torch.no_grad():
                            _ = compiled_model.generate([test_entries], [test_attributes])
                        
                        tts_model = compiled_model
                        COMPILATION_ENABLED = True
                        logger.info("✅ Strategy 3 SUCCESS! Model compiled with max-autotune-no-cudagraphs!")
                        
                    except Exception as compile_error_3:
                        logger.warning(f"⚠️  Strategy 3 failed: {str(compile_error_3)}")
                        logger.debug(f"Strategy 3 error details: {type(compile_error_3).__name__}: {compile_error_3}")
                        
                        # Strategy 4: Try with inductor backend
                        try:
                            logger.info("📈 Strategy 4: Compilation with inductor backend...")
                            compiled_model = torch.compile(
                                tts_model, 
                                mode="reduce-overhead",  # Aggressive mode
                                fullgraph=False,  # Allow graph breaks
                                dynamic=True,     # Handle variable inputs
                                backend="inductor"  # Use inductor backend
                            )
                            
                            # Test the compiled model
                            logger.info("🧪 Testing compiled model...")
                            test_entries = tts_model.prepare_script(["test"], padding_between=1)
                            test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                            test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                            
                            with torch.no_grad():
                                _ = compiled_model.generate([test_entries], [test_attributes])
                            
                            tts_model = compiled_model
                            COMPILATION_ENABLED = True
                            logger.info("✅ Strategy 4 SUCCESS! Model compiled with inductor backend!")
                            
                        except Exception as compile_error_4:
                            logger.warning(f"⚠️  Strategy 4 failed: {str(compile_error_4)}")
                            logger.debug(f"Strategy 4 error details: {type(compile_error_4).__name__}: {compile_error_4}")
                            
                            # Strategy 5: Try with mixed precision
                            try:
                                logger.info("📈 Strategy 5: Compilation with mixed precision...")
                                
                                # Enable mixed precision
                                torch.set_autocast_enabled(True)
                                
                                compiled_model = torch.compile(
                                    tts_model, 
                                    mode="reduce-overhead",  # Aggressive mode
                                    fullgraph=False,  # Allow graph breaks
                                    dynamic=True,     # Handle variable inputs
                                    backend="inductor"  # Use inductor backend
                                )
                                
                                # Test the compiled model with autocast
                                logger.info("🧪 Testing compiled model with mixed precision...")
                                test_entries = tts_model.prepare_script(["test"], padding_between=1)
                                test_voice_path = tts_model.get_voice_path(VOICE_OPTIONS[DEFAULT_VOICE])
                                test_attributes = tts_model.make_condition_attributes([test_voice_path], cfg_coef=2.5)
                                
                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    with torch.no_grad():
                                        _ = compiled_model.generate([test_entries], [test_attributes])
                                
                                tts_model = compiled_model
                                COMPILATION_ENABLED = True
                                logger.info("✅ Strategy 5 SUCCESS! Model compiled with mixed precision!")
                                
                            except Exception as compile_error_5:
                                logger.warning(f"⚠️  Strategy 5 failed: {str(compile_error_5)}")
                                logger.debug(f"Strategy 5 error details: {type(compile_error_5).__name__}: {compile_error_5}")
                                
                                # All strategies failed
                                logger.warning("❌ All compilation strategies failed. Using uncompiled model.")
                                COMPILATION_ENABLED = False
                                
                                # Detailed error analysis
                                all_errors = [compile_error_1, compile_error_2, compile_error_3, compile_error_4, compile_error_5]
                                error_summary = "\n".join([f"Strategy {i+1}: {type(e).__name__}: {str(e)}" for i, e in enumerate(all_errors)])
                                logger.debug(f"Compilation error summary:\n{error_summary}")
                                
                                # Common error patterns
                                if any("scatter" in str(e).lower() for e in all_errors):
                                    logger.info("💡 Common issue: Scatter operations not supported in compilation")
                                elif any("dtype" in str(e).lower() for e in all_errors):
                                    logger.info("💡 Common issue: Data type mismatches in compilation")
                                elif any("graph break" in str(e).lower() for e in all_errors):
                                    logger.info("💡 Common issue: Graph breaks prevent compilation")
                                elif any("memory" in str(e).lower() for e in all_errors):
                                    logger.info("💡 Common issue: Memory allocation problems during compilation")
                                elif any("inductor" in str(e).lower() for e in all_errors):
                                    logger.info("💡 Common issue: Inductor backend not available or compatible")
                                else:
                                    logger.info("💡 Unknown compilation issues. The uncompiled model will work fine.")
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
                                reduce_noise: bool = True, fast_mode: bool = False, 
                                filename: str = None) -> str:
    """Generate audio with maximum GPU utilization through batching and concurrency"""
    global ENABLE_BATCH_PROCESSING  # Fix scoping issue
    
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded")

    start_time = time.time()
    
    # Generate output filename
    if filename:
        # Use custom filename with proper extension
        if not filename.endswith(f'.{output_format}'):
            filename = f"{filename}.{output_format}"
        output_filename = filename
        logger.info(f"📁 Using custom filename: {output_filename}")
    else:
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_filename = f"tts_output_{timestamp}.{output_format}"
        logger.info(f"📁 Generated filename: {output_filename}")
    
    # Fast mode optimizes for speed while keeping audio cleaning for quality
    if fast_mode:
        logger.info("🚀 Starting FAST MODE audio generation (optimized settings with audio cleaning)...")
        # Fast mode keeps audio cleaning but uses optimized settings
        # - Larger batch sizes
        # - More aggressive CUDA settings
        # - Better memory management
        # - Audio cleaning for quality
        
        # Use larger batch size for fast mode
        fast_batch_size = min(24, MAX_BATCH_SIZE * 2)  # Triple the batch size for fast mode
        logger.info(f"🚀 Fast mode using batch size: {fast_batch_size}")
        
        # Enable more aggressive optimizations for fast mode
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Pre-allocate more memory for fast mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
    else:
        logger.info("🎵 Starting HIGH GPU UTILIZATION audio generation...")
        fast_batch_size = MAX_BATCH_SIZE
    
    # Enable mixed precision for compiled models if available
    use_mixed_precision = COMPILATION_ENABLED and torch.cuda.is_available()
    if use_mixed_precision:
        logger.info("🚀 Using mixed precision for compiled model optimization")
    
    # Use memory pool context for automatic cleanup
    with MemoryPoolContext() as memory_pool:
        
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
        
        # Pre-allocate GPU memory for batch processing using memory pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Pre-allocate larger memory chunks for batch processing
            try:
                if gpu_memory_pool:
                    # Pre-allocate batch-sized tensors
                    batch_size = fast_batch_size * 1024
                    dummy_tensor = allocate_gpu_tensor(batch_size, torch.float32, device)
                    free_gpu_tensor(dummy_tensor)
                else:
                    dummy_tensor = torch.zeros(fast_batch_size, 1024, device=device)
                    del dummy_tensor
            except:
                pass  # Continue if allocation fails
        
        logger.info(f"🚀 Processing {len(text_segments)} text segments in batches of {fast_batch_size}")
        
        # Process text segments in batches for higher GPU utilization
        text_audio_results = {}
        batch_processing_enabled = ENABLE_BATCH_PROCESSING  # Local copy to avoid scoping issues
        
        if batch_processing_enabled and len(text_segments) > 1:
            # BATCH PROCESSING MODE - Higher GPU Utilization
            for batch_start in range(0, len(text_segments), fast_batch_size):
                batch_end = min(batch_start + fast_batch_size, len(text_segments))
                batch = text_segments[batch_start:batch_end]
                
                logger.info(f"🔥 Processing batch {batch_start//fast_batch_size + 1} with {len(batch)} segments (GPU INTENSIVE)")
                
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
                    # Initialize batch_results outside conditional blocks
                    batch_results = []
                    
                    # Use mixed precision context for compiled models
                    if use_mixed_precision:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            with torch.no_grad():
                                # Create multiple CUDA streams for concurrent execution
                                if torch.cuda.is_available():
                                    # Use more streams for fast mode
                                    max_streams = 16 if fast_mode else 8
                                    streams = [torch.cuda.Stream() for _ in range(min(len(batch), max_streams))]
                                    
                                    # Launch parallel generations
                                    for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                                        stream_idx = i % len(streams)  # Cycle through available streams
                                        with torch.cuda.stream(streams[stream_idx]):
                                            with torch.no_grad():
                                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                                    result = tts_model.generate([entries], [attributes])
                                            
                                            # Store result object directly (not audio[0])
                                            batch_results.append((batch_indices[i], result))
                                    
                                    # Synchronize all streams
                                    torch.cuda.synchronize()
                                else:
                                    # CPU fallback
                                    for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                                        result = tts_model.generate([entries], [attributes])
                                        batch_results.append((batch_indices[i], result))
                    else:
                        # Standard precision for uncompiled models
                        with torch.no_grad():
                            # Create multiple CUDA streams for concurrent execution
                            if torch.cuda.is_available():
                                # Use more streams for fast mode
                                max_streams = 16 if fast_mode else 8
                                streams = [torch.cuda.Stream() for _ in range(min(len(batch), max_streams))]
                                
                                # Launch parallel generations
                                for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                                    stream_idx = i % len(streams)  # Cycle through available streams
                                    with torch.cuda.stream(streams[stream_idx]):
                                        with torch.no_grad():
                                            result = tts_model.generate([entries], [attributes])
                                    
                                    # Store result object directly (not audio[0])
                                    batch_results.append((batch_indices[i], result))
                                
                                # Synchronize all streams
                                torch.cuda.synchronize()
                            else:
                                # CPU fallback
                                for i, (entries, attributes) in enumerate(zip(batch_entries, batch_attributes)):
                                    result = tts_model.generate([entries], [attributes])
                                    batch_results.append((batch_indices[i], result))
                    
                    logger.info(f"✅ Batch completed with {len(batch_results)} parallel generations")
                    
                    # Decode all results (keeping on GPU as long as possible)
                    for idx, result in batch_results:
                        try:
                            # Use mixed precision for decoding if compiled
                            if use_mixed_precision:
                                with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                            else:
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
                    # Use mixed precision for compiled models
                    if use_mixed_precision:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                    else:
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
                        temp_wav_path = wav_fp.name
                        logger.info(f"📁 Creating temporary WAV file: {temp_wav_path}")
                        sphn.write_wav(temp_wav_path, pcm_data, tts_model.mimi.sample_rate)
                        segment_audio = AudioSegment.from_wav(temp_wav_path)
                        output_audio_segments.append(segment_audio)
                        os.unlink(temp_wav_path)
                        logger.info(f"📁 Cleaned up temporary file: {temp_wav_path}")
        
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
                logger.info("🧹 Applying GPU-accelerated cleaning...")
                
                # Use standard cleaning for now to avoid dimension issues
                final_audio = clean_audio_segment_gpu(
                    final_audio, 
                    volume_boost, 
                    remove_crackles, 
                    apply_filters, 
                    reduce_noise
                )
                logger.info(f"🧹 GPU cleaning completed in {time.time() - cleaning_start:.2f}s")
            except Exception as cleaning_error:
                logger.warning(f"GPU cleaning failed, using standard cleaning: {cleaning_error}")
                # Fallback to standard cleaning
                try:
                    final_audio = clean_audio_segment_gpu(
                        final_audio, 
                        volume_boost, 
                        remove_crackles, 
                        apply_filters, 
                        reduce_noise
                    )
                except Exception as fallback_error:
                    logger.error(f"Audio cleaning failed: {fallback_error}")
                    # Continue without cleaning
        else:
            # Standard GPU cleaning
            try:
                logger.info("🧹 Applying standard GPU cleaning...")
                final_audio = clean_audio_segment_gpu(final_audio, volume_boost, remove_crackles, apply_filters, reduce_noise)
            except Exception as cleaning_error:
                logger.warning(f"GPU cleaning failed: {cleaning_error}")
        
        # Export
        export_start = time.time()
        if output_format.lower() == "wav":
            # Use custom filename or generate temporary one
            if filename:
                output_path = f"/tmp/{output_filename}"
                logger.info(f"📁 Exporting WAV to custom path: {output_path}")
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_fp:
                    output_path = wav_fp.name
                    logger.info(f"📁 Exporting WAV to temporary file: {output_path}")
            
            # Export the audio
            final_audio.export(output_path, format="wav", parameters=["-ac", "1"])
            export_time = time.time() - export_start
            logger.info(f"📁 Final audio exported as WAV in {export_time:.2f}s")
            logger.info(f"📁 Output file saved to: {output_path}")
            total_time = time.time() - start_time
            logger.info(f"🎉 HIGH GPU UTILIZATION generation completed in {total_time:.2f}s")
            if ENABLE_MONITORING:
                log_system_usage()
            return output_path
        else:
            # Use custom filename or generate temporary one
            if filename:
                output_path = f"/tmp/{output_filename}"
                logger.info(f"📁 Exporting MP3 to custom path: {output_path}")
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_fp:
                    output_path = mp3_fp.name
                    logger.info(f"📁 Exporting MP3 to temporary file: {output_path}")
            
            # Export the audio
            final_audio.export(output_path, format="mp3", bitrate="320k", parameters=["-ac", "1"])
            export_time = time.time() - export_start
            logger.info(f"📁 Final audio exported as MP3 in {export_time:.2f}s")
            logger.info(f"📁 Output file saved to: {output_path}")
            total_time = time.time() - start_time
            logger.info(f"🎉 HIGH GPU UTILIZATION generation completed in {total_time:.2f}s")
            if ENABLE_MONITORING:
                log_system_usage()
            return output_path

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
    fast_mode: bool = False  # Skip audio cleaning for maximum speed
    filename: str = None  # Allow custom filename for output

class AudioCleaningRequest(BaseModel):
    volume_boost: float = 6.0
    remove_crackles: bool = True
    apply_filters: bool = True
    reduce_noise: bool = True

# --- API Endpoints (updated for GPU optimization) ---
@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    """Generate TTS audio with optional GPU-accelerated cleaning and caching"""
    logger.info(f"TTS API endpoint hit. Format: {request.output_format}, Cleaning: {request.apply_cleaning}")
    
    if request.output_format.lower() not in ["mp3", "wav"]:
        return JSONResponse(
            status_code=400, 
            content={"detail": "Invalid output format. Use 'mp3' or 'wav'."}
        )
    
    # Generate cache key
    cache_key = audio_cache.generate_cache_key(
        request.text, 
        request.voice_choice, 
        request.output_format,
        apply_cleaning=request.apply_cleaning,
        volume_boost=request.volume_boost,
        remove_crackles=request.remove_crackles,
        apply_filters=request.apply_filters,
        reduce_noise=request.reduce_noise
    )
    
    # Check cache first
    cached_path = audio_cache.get_cached_audio(cache_key)
    if cached_path:
        if request.output_format.lower() == "wav":
            media_type = "audio/wav"
            filename = request.filename if request.filename else "generated_speech.wav"
        else:
            media_type = "audio/mpeg"
            filename = request.filename if request.filename else "generated_speech.mp3"
        
        return FileResponse(cached_path, media_type=media_type, filename=filename)
    
    # Generate new audio through request queue
    async def generate_audio_async():
        return generate_audio(
            request.text, 
            request.voice_choice, 
            output_format=request.output_format,
            apply_cleaning=request.apply_cleaning,
            volume_boost=request.volume_boost,
            remove_crackles=request.remove_crackles,
            apply_filters=request.apply_filters,
            reduce_noise=request.reduce_noise,
            fast_mode=request.fast_mode,
            filename=request.filename
        )
    
    try:
        output_path = await request_queue.submit_request(generate_audio_async)
        
        # Cache the result
        audio_cache.cache_audio(cache_key, output_path, {
            "text_length": len(request.text),
            "voice": request.voice_choice,
            "format": request.output_format
        })
        
        if request.output_format.lower() == "wav":
            media_type = "audio/wav"
            filename = request.filename if request.filename else "generated_speech.wav"
        else:
            media_type = "audio/mpeg"
            filename = request.filename if request.filename else "generated_speech.mp3"
        
        return FileResponse(output_path, media_type=media_type, filename=filename)
        
    except RuntimeError as e:
        if "maximum capacity" in str(e):
            return JSONResponse(
                status_code=503, 
                content={"detail": "Server is at maximum capacity. Please try again later."}
            )
        else:
            return JSONResponse(status_code=400, content={"detail": str(e)})
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

@app.post("/api/memory/cleanup")
def cleanup_memory():
    """Force cleanup of GPU memory pool"""
    if gpu_memory_pool:
        gpu_memory_pool.cleanup(force=True)
        return {"status": "success", "message": "GPU memory pool cleaned"}
    else:
        return {"status": "no_op", "message": "No GPU memory pool available"}

@app.get("/api/memory/stats")
def get_memory_stats():
    """Get detailed memory pool statistics"""
    if gpu_memory_pool:
        stats = gpu_memory_pool.get_memory_stats()
        return {
            "status": "success",
            "memory_pool": stats,
            "pytorch_memory": {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "cached_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            }
        }
    else:
        return {"status": "no_pool", "message": "No GPU memory pool available"}

@app.get("/api/queue/stats")
def get_queue_stats():
    """Get request queue statistics"""
    return {
        "status": "success",
        "request_queue": request_queue.get_stats(),
        "audio_cache": audio_cache.get_stats()
    }

@app.post("/api/cache/clear")
def clear_cache():
    """Clear the audio cache"""
    audio_cache.cache.clear()
    audio_cache.cache_stats = {
        "hits": 0,
        "misses": 0,
        "evictions": 0
    }
    return {"status": "success", "message": "Audio cache cleared"}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    memory_pool_stats = gpu_memory_pool.get_memory_stats() if gpu_memory_pool else None
    queue_stats = request_queue.get_stats()
    cache_stats = audio_cache.get_stats()
    
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "compilation_enabled": COMPILATION_ENABLED if MODEL_LOADED else False,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
        "memory_pool": memory_pool_stats,
        "request_queue": queue_stats,
        "audio_cache": cache_stats,
        "features": ["tts", "gpu_audio_cleaning", "ssml_support", "gpu_acceleration", "error_recovery", "memory_pooling", "request_queue", "audio_caching"]
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
            "health": "/api/health",
            "memory_stats": "/api/memory/stats",
            "memory_cleanup": "/api/memory/cleanup",
            "queue_stats": "/api/queue/stats",
            "cache_clear": "/api/cache/clear"
        },
        "features": [
            "GPU-accelerated Text-to-Speech with multiple voices and formats (MP3/WAV)",
            "SSML support",
            "GPU-accelerated audio cleaning (crackle removal, noise reduction)",
            "Hardware-optimized volume boosting and filtering",
            "Multiple audio format support for file cleaning",
            "Automatic mixed precision for improved performance",
            "Advanced GPU memory pooling for efficient memory management",
            "Request queue management for load balancing",
            "Audio caching for improved response times",
            "Adaptive memory management based on usage patterns"
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