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
import uvicorn
import logging
import re
import os
import html
from xml.etree import ElementTree as ET

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(title="Kyutai TTS API", description="Generate downloadable speech from text.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Model Loading ---
MODEL_LOADED = False
tts_model = None
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
    logger.info("Model loaded successfully.")
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

def generate_audio(ssml_text: str, default_voice: str) -> str:
    """
    Generates an MP3 audio file from an SSML or plain text string.
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

    final_audio = final_audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_fp:
        final_audio.export(mp3_fp.name, format="mp3", bitrate="320k")
        logger.info(f"Final audio successfully exported to {mp3_fp.name}")
        return mp3_fp.name

# --- API Endpoint ---
class TTSRequest(BaseModel):
    text: str
    voice_choice: str = DEFAULT_VOICE

@app.post("/api/tts")
def tts_endpoint(request: TTSRequest):
    """
    Handles TTS requests. Returns an audio file on success or a JSON error on failure.
    """
    logger.info("TTS API endpoint hit.")
    try:
        output_path = generate_audio(request.text, request.voice_choice)
        return FileResponse(output_path, media_type="audio/mpeg", filename="generated_speech.mp3")
    except ValueError as e:
        logger.error(f"Invalid input provided: {e}")
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        logger.exception("An internal error occurred during audio generation.")
        return JSONResponse(status_code=500, content={"detail": f"An internal error occurred: {str(e)}"})

# --- Main Execution ---
if __name__ == "__main__":
    if MODEL_LOADED:
        logger.info("Launching FastAPI server on http://0.0.0.0:7861")
        uvicorn.run(app, host="0.0.0.0", port=7861)
    else:
        logger.error("Application will not start because the model failed to load.")
