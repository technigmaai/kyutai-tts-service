"""
SSML parsing utilities and audio format helpers
"""

import logging
import re
import html
from xml.etree import ElementTree as ET
from pydub import AudioSegment
from config import VOICE_OPTIONS

logger = logging.getLogger(__name__)

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
            ms = parse_break_duration(duration_str)
            silence = AudioSegment.silent(duration=ms)
            segments.append(("PAUSE", silence))
            
    logger.info(f"Parsed into {len(segments)} segments.")
    return segments

def parse_break_duration(duration_str: str) -> int:
    """
    Parse an SSML <break> time attribute into milliseconds.

    Supports the SSML time formats: "500ms", "1s", "1.5s", "10s".
    Defaults to 500ms if the value cannot be parsed.
    """
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*$", duration_str, re.IGNORECASE)
    if not match:
        logger.warning(f"Could not parse break duration '{duration_str}', defaulting to 500ms")
        return 500
    value = float(match.group(1))
    unit = (match.group(2) or "ms").lower()
    if unit == "s":
        return int(value * 1000)
    return int(value)


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