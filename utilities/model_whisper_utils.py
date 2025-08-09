import time
from pathlib import Path
from typing import Optional

import torch
from faster_whisper import WhisperModel
from loguru import logger

from .file_utils import (get_transcription_filepath_by_hash,
                         save_transcription_to_cache)

#WHISPER_MODEL = "Numbat/faster-skyrim-whisper-base.en"
WHISPER_MODEL = "distil-whisper/distil-large-v3.5-ct2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_ENGINE: Optional[WhisperModel]= None

def initialize_whisper_model():
    global  WHISPER_ENGINE
    
    if WHISPER_ENGINE is None:
            # Initialize Whisper model
            logger.info("Loading Whisper model...")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="bfloat16")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="float16" if DEVICE=="cuda" else "int8")
            logger.info("Whisper model loaded successfully.")
    return WHISPER_ENGINE


def transcribe_audio_with_whisper(audio_path: str, audio_file_hash: str ,language: str = "en") -> str:
    """Transcribe audio using local whispercpp server with caching."""

    filename = Path(audio_path).stem
    # Check cache first
    cached_transcription = get_transcription_filepath_by_hash(filename, audio_file_hash)
    if cached_transcription:
        logger.info(f"Using cached transcription for {filename} with {audio_file_hash}")
        return cached_transcription.read_text(encoding="utf-8")

    logger.info(f"Transcribing audio file: {audio_path}")
    start_time = time.time()

    try:

        texts = []
        segments, info = WHISPER_ENGINE.transcribe(audio_path, beam_size=2, vad_filter=True,without_timestamps= True,multilingual= True)
        for segment in segments:
          #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
          texts.append(segment.text.strip())
        transcription = ' '.join(texts).strip()

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"Transcription completed in {elapsed_ms:.2f}ms: '{transcription}'")

        # Save to cache
        if transcription:
            save_transcription_to_cache(filename, audio_file_hash, transcription)
            logger.info(f"Saved transcription to cache: {audio_file_hash}")

        return transcription

    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        # Fallback to a generic message if transcription fails
        return "This is the voice you should use for the generation."