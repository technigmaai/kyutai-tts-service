# 🎵 Real ClearerVoice-Studio Integration Guide

This document explains how to use the **real ClearerVoice-Studio** for professional-grade audio enhancement in the TTS service.

## 📚 Overview

ClearerVoice-Studio is a state-of-the-art audio enhancement system that provides superior audio quality compared to basic GPU cleaning. This integration uses the **actual ClearerVoice-Studio implementation** from the [official repository](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice).

## 🚀 Features

### **Real ClearerVoice-Studio Integration**
- **Official Implementation**: Uses the actual ClearerVoice-Studio code
- **Repository Integration**: Direct integration with the official repository
- **Superior Quality**: Professional-grade audio enhancement
- **Advanced Algorithms**: State-of-the-art audio processing

### **Integration Benefits**
- **Real Implementation**: Uses the actual ClearerVoice-Studio codebase
- **Fallback Support**: Automatically falls back to modelscope or GPU cleaning
- **Flexible Usage**: Can be used independently or with GPU cleaning
- **Quality Options**: Choose between real ClearerVoice and GPU cleaning

## 📋 Installation

### **1. Install Real ClearerVoice-Studio**
```bash
# Run the real ClearerVoice installation script
./install_real_clearvoice.sh
```

### **2. Verify Installation**
```bash
# Check if real ClearerVoice is available
curl "http://localhost:7861/api/health" | jq '.clearvoice_info.clearvoice_type'
```

## 🎯 Usage

### **API Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_clearvoice` | boolean | true | Use real ClearerVoice-Studio for enhancement (if available) |
| `clearvoice_enhancement` | boolean | false | Enable real ClearerVoice enhancement (overrides apply_cleaning) |

### **TTS with Real ClearerVoice Enhancement**

#### **Real ClearerVoice Enhancement**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses real ClearerVoice-Studio for professional enhancement.",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "output_format": "wav"
  }' \
  --output enhanced_audio.wav
```

#### **Real ClearerVoice + GPU Cleaning Hybrid**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This audio uses both real ClearerVoice and GPU cleaning.",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "apply_cleaning": true,
    "output_format": "mp3"
  }' \
  --output hybrid_enhanced.mp3
```

### **File Enhancement Endpoint**

#### **Enhance Existing Audio Files with Real ClearerVoice**
```bash
curl -X POST "http://localhost:7861/api/clearvoice-enhance" \
  -F "audio_file=@input_audio.mp3" \
  -F "use_clearvoice=true" \
  --output enhanced_output.mp3
```

## 🔧 Configuration

### **Enhancement Priority**

1. **Real ClearerVoice Enhancement** (if `clearvoice_enhancement: true`)
   - Uses the actual ClearerVoice-Studio implementation
   - Overrides GPU cleaning settings
   - Highest quality output

2. **Modelscope Fallback** (if real ClearerVoice not available)
   - Uses modelscope ClearerVoice model
   - Good quality with reliable operation

3. **GPU Cleaning** (if both ClearerVoice options not available)
   - Uses GPU-accelerated cleaning
   - Configurable parameters
   - Good quality with fast processing

4. **No Enhancement** (if all disabled)
   - Raw TTS output
   - Fastest processing
   - No quality improvements

### **Quality Comparison**

| Method | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| **Real ClearerVoice** | 🎵 Professional | ⚡ Medium | High-quality applications |
| **Modelscope Fallback** | 🎵 Good | ⚡ Medium | Reliable enhancement |
| **GPU Cleaning** | 🎵 Good | ⚡ Fast | Balanced quality/speed |
| **No Enhancement** | 🎵 Raw | ⚡ Fastest | Real-time applications |

## 📊 Performance Characteristics

### **Real ClearerVoice-Studio**
- **Processing Time**: 3-8 seconds per audio segment
- **Memory Usage**: ~2-4GB GPU memory
- **Quality**: Professional studio-grade
- **Implementation**: Official repository code

### **Modelscope Fallback**
- **Processing Time**: 2-5 seconds per audio segment
- **Memory Usage**: ~2-4GB GPU memory
- **Quality**: Professional-grade
- **Model Size**: ~500MB (downloaded automatically)

### **GPU Cleaning**
- **Processing Time**: 0.5-2 seconds per audio segment
- **Memory Usage**: ~1-2GB GPU memory
- **Quality**: Good to very good
- **Model Size**: No additional models required

## 🐛 Troubleshooting

### **Common Issues**

#### **1. Real ClearerVoice Not Available**
```bash
# Check if repository is cloned
ls -la clearvoice_studio/

# Reinstall real ClearerVoice
./install_real_clearvoice.sh
```

#### **2. Repository Download Issues**
```bash
# Clear repository and reinstall
rm -rf clearvoice_studio
./install_real_clearvoice.sh
```

#### **3. Import Issues**
```bash
# Check Python path
python3 -c "import sys; print('\\n'.join(sys.path))"

# Test ClearerVoice import
python3 -c "
import sys
sys.path.insert(0, './clearvoice_studio/clearvoice')
try:
    from clearvoice import ClearerVoice
    print('✅ Real ClearerVoice-Studio is available')
except ImportError as e:
    print(f'❌ ClearerVoice-Studio not available: {e}')
"
```

### **Error Messages**

| Error | Solution |
|-------|----------|
| `Repository not found` | Check internet connection and git installation |
| `Import error` | Run `./install_real_clearvoice.sh` |
| `CUDA out of memory` | Use GPU cleaning or reduce batch size |
| `Processing timeout` | Use GPU cleaning for faster processing |

## 🔍 Monitoring

### **Health Check**
```bash
# Check ClearerVoice availability and type
curl "http://localhost:7861/api/health" | jq '.clearvoice_info'
```

### **Enhancement Info**
```bash
# Get detailed enhancement information
curl "http://localhost:7861/api/health" | jq '.clearvoice_info.clearvoice_type'
```

## 📈 Best Practices

### **When to Use Real ClearerVoice**
- **High-quality applications**: Podcasts, audiobooks, professional content
- **Official implementation needed**: When you need the real ClearerVoice-Studio
- **Maximum quality**: When quality is more important than speed
- **Professional output**: Commercial applications

### **When to Use Modelscope Fallback**
- **Reliable operation**: When real ClearerVoice has issues
- **Good quality needed**: Professional enhancement without complexity
- **Stable performance**: Consistent results

### **When to Use GPU Cleaning**
- **Real-time applications**: Live streaming, interactive systems
- **Speed critical**: High-volume processing
- **Good input quality**: Clean TTS output
- **Resource constrained**: Limited GPU memory

### **When to Use No Enhancement**
- **Maximum speed**: Real-time applications
- **Raw quality needed**: Research, analysis
- **Minimal processing**: Simple applications

## 🎯 Examples

### **Professional Podcast with Real ClearerVoice**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to our professional podcast with real ClearerVoice-Studio enhancement.",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "output_format": "wav",
    "filename": "podcast_real_clearvoice.wav"
  }' \
  --output podcast_real_clearvoice.wav
```

### **Educational Content with Fallback**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This educational content uses ClearerVoice with automatic fallback.",
    "clearvoice_enhancement": true,
    "use_clearvoice": true,
    "output_format": "mp3",
    "filename": "educational_clearvoice.mp3"
  }' \
  --output educational_clearvoice.mp3
```

### **Fast Processing with GPU Cleaning**
```bash
curl -X POST "http://localhost:7861/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Quick processing for real-time applications.",
    "apply_cleaning": true,
    "clearvoice_enhancement": false,
    "fast_mode": true,
    "output_format": "mp3"
  }' \
  --output fast_output.mp3
```

## 📚 References

- **[ClearerVoice-Studio Repository](https://github.com/modelscope/ClearerVoice-Studio)**: Official implementation
- **[ClearerVoice Module](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice)**: Main enhancement module
- **[ModelScope Documentation](https://modelscope.cn/)**: Model hosting platform

---

**Last Updated**: July 23, 2025  
**Version**: 2.0.0  
**Real ClearerVoice Integration**: ✅  
**Professional Quality**: ✅  
**Official Implementation**: ✅ 