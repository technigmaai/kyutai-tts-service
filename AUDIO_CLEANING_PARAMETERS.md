# 🧹 Audio Cleaning Parameters - How They Work

This document explains how the audio cleaning parameters work in the GPU-Optimized Kyutai TTS Service.

## **Overview**

The TTS service provides sophisticated GPU-accelerated audio cleaning capabilities. The cleaning system uses a **master switch** approach where `apply_cleaning` controls whether cleaning is enabled, and individual parameters control specific cleaning steps.

## **🎛️ Parameter Hierarchy**

### **Master Switch: `apply_cleaning`**
- **Purpose**: Enables/disables the entire audio cleaning pipeline
- **Default**: `false` (cleaning is OFF by default)
- **When `true`**: The cleaning pipeline runs with individual parameter settings
- **When `false`**: No cleaning is applied, regardless of individual parameter settings

### **Individual Control Parameters**
When `apply_cleaning` is `true`, these parameters control specific cleaning steps:

| Parameter | Controls | Default | Description |
|-----------|----------|---------|-------------|
| `remove_crackles` | Crackle/pop removal | `true` | Removes audio artifacts and clicks |
| `apply_filters` | High/low pass filters | `true` | Applies frequency filtering |
| `reduce_noise` | Spectral noise reduction | `true` | Reduces background noise |
| `volume_boost` | Volume enhancement | `6.0` dB | Increases audio volume |

## **🧹 Cleaning Pipeline Steps**

When `apply_cleaning` is enabled, the GPU-accelerated pipeline runs these steps **in order**:

### **1. Basic Cleaning (Always Runs)**
```python
self.remove_dc_offset()  # Remove DC offset
```

### **2. Conditional Steps (Based on Parameters)**
```python
if remove_crackles:
    self.remove_clicks_pops_gpu()  # Remove audio artifacts

if apply_filters:
    self.apply_highpass_filter_gpu(cutoff=80)    # Remove low frequencies
    self.apply_lowpass_filter_gpu(cutoff=8000)   # Remove high frequencies

if reduce_noise:
    self.reduce_noise_spectral_gpu()  # Spectral noise reduction
```

### **3. Volume and Dynamics (Always Runs)**
```python
self.normalize_audio_gpu()           # Normalize audio levels
self.apply_gentle_compression_gpu()  # Apply gentle compression

if volume_boost_db > 0:
    self.boost_volume_gpu(volume_boost_db)  # Boost volume
```

## **📋 Usage Examples**

### **Example 1: No Cleaning (Fastest)**
```json
{
    "text": "Hello world",
    "apply_cleaning": false
    // Individual parameters are ignored
}
```
**Result**: Raw TTS output, no processing overhead

### **Example 2: Full Cleaning (Highest Quality)**
```json
{
    "text": "Hello world",
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true,
    "volume_boost": 6.0
}
```
**Result**: Complete audio cleaning pipeline applied

### **Example 3: Partial Cleaning (Balanced)**
```json
{
    "text": "Hello world",
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": false,  // Skip filters
    "reduce_noise": false,   // Skip noise reduction
    "volume_boost": 3.0      // Less volume boost
}
```
**Result**: Only crackle removal and moderate volume boost

### **Example 4: Only Volume Boost**
```json
{
    "text": "Hello world",
    "apply_cleaning": true,
    "remove_crackles": false,
    "apply_filters": false,
    "reduce_noise": false,
    "volume_boost": 8.0      // Only volume boost
}
```
**Result**: Only volume enhancement applied

### **Example 5: SSML with Cleaning**
```json
{
    "text": "<speak><voice name=\"Happy\">Hello!</voice><break time=\"1s\"/><voice name=\"Calm\">This is clean audio.</voice></speak>",
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true,
    "volume_boost": 6.0
}
```
**Result**: Multi-voice SSML with full cleaning applied

## **🎯 Key Points**

### **Parameter Dependencies**
1. **`apply_cleaning` is the master switch** - must be `true` for any cleaning to happen
2. **Individual parameters only matter when `apply_cleaning` is `true`**
3. **You can selectively enable/disable specific cleaning steps**
4. **Some steps always run** (DC offset removal, normalization, compression)
5. **Volume boost only applies when `apply_cleaning` is `true`**

### **Performance Characteristics**
- **`apply_cleaning: false`**: Fastest, no cleaning overhead
- **`apply_cleaning: true`**: Slower but higher quality audio
- **More cleaning steps = more processing time**
- **GPU acceleration makes cleaning much faster than CPU**

### **Quality vs Speed Trade-offs**
| Configuration | Speed | Quality | Use Case |
|--------------|-------|---------|----------|
| No cleaning | ⚡ Fastest | 🎵 Raw | Real-time applications |
| Volume only | ⚡ Fast | 🎵 Good | Simple enhancement |
| Partial cleaning | ⚡ Fast | 🎵 Better | Balanced approach |
| Full cleaning | ⚡ Slower | 🎵 Best | High-quality output |

## **🔧 Technical Details**

### **GPU-Accelerated Processing**
- **Memory Pool**: Efficient tensor allocation and reuse
- **Batch Processing**: Multiple audio segments processed simultaneously
- **Mixed Precision**: Automatic FP16/FP32 optimization
- **Concurrent Operations**: Multiple GPU streams for parallel processing

### **Audio Processing Steps**
1. **DC Offset Removal**: Eliminates DC bias
2. **Crackle Removal**: Detects and removes audio artifacts
3. **High-Pass Filter**: Removes frequencies below 80 Hz
4. **Low-Pass Filter**: Removes frequencies above 8 kHz
5. **Spectral Noise Reduction**: Reduces background noise
6. **Normalization**: Ensures consistent audio levels
7. **Compression**: Applies gentle dynamic compression
8. **Volume Boost**: Increases overall volume

### **Error Handling**
- **Graceful Fallback**: If GPU cleaning fails, falls back to CPU processing
- **Partial Failure**: If specific steps fail, continues with remaining steps
- **No Data Loss**: Original audio preserved if cleaning fails completely

## **📊 Monitoring and Debugging**

### **Health Check Endpoint**
```bash
curl "http://localhost:7861/api/health"
```
Returns cleaning pipeline status and GPU memory usage.

### **Memory Statistics**
```bash
curl "http://localhost:7861/api/memory/stats"
```
Shows GPU memory pool utilization and cleaning performance metrics.

### **Queue Statistics**
```bash
curl "http://localhost:7861/api/queue/stats"
```
Displays request queue status and processing times.

## **🚀 Best Practices**

### **For Real-time Applications**
```json
{
    "apply_cleaning": false
}
```

### **For High-Quality Output**
```json
{
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true,
    "volume_boost": 6.0
}
```

### **For Balanced Performance**
```json
{
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": false,
    "reduce_noise": false,
    "volume_boost": 3.0
}
```

### **For Podcast/Audio Content**
```json
{
    "apply_cleaning": true,
    "remove_crackles": true,
    "apply_filters": true,
    "reduce_noise": true,
    "volume_boost": 8.0
}
```

## **⚠️ Troubleshooting**

### **Common Issues**

**Cleaning Not Applied**
- Check that `apply_cleaning` is set to `true`
- Verify GPU is available and working
- Check logs for error messages

**Poor Audio Quality**
- Enable more cleaning steps
- Increase `volume_boost`
- Check GPU memory availability

**Slow Performance**
- Disable unnecessary cleaning steps
- Use `fast_mode: true` for speed optimization
- Monitor GPU memory usage

**GPU Memory Errors**
- Reduce batch size
- Clear GPU memory cache
- Restart the service

---

**Last Updated**: July 21, 2025  
**Version**: 1.0.0  
**GPU Optimized**: ✅  
**Production Ready**: ✅ 