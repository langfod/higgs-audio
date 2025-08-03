import time
from typing import Optional

from faster_whisper import WhisperModel
from loguru import logger

from .file_utils import get_file_hash, save_transcription_to_cache, get_transcription_filepath_by_hash

WHISPER_API_URL = "http://localhost:8080/inference"
WHISPER_MODEL = "distil-whisper/distil-large-v3.5-ct2"

WHISPER_ENGINE: Optional[WhisperModel]= None

def initialize_whisper_model():
    global  WHISPER_ENGINE

    if WHISPER_ENGINE is None:
        try:
            # Initialize Whisper model
            logger.info("Loading Whisper model...")
            WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
    return WHISPER_ENGINE


def transcribe_audio_with_whisper(audio_path: str) -> str:
    """Transcribe audio using local whispercpp server with caching."""

    # Get md5 hash for caching
    audio_hash = get_file_hash(audio_path)
    logger.info(f"Audio file hash: {audio_hash}")

    # Check cache first
    cached_transcription = get_transcription_filepath_by_hash(audio_hash)
    if cached_transcription:
        logger.info(f"Using cached transcription for {audio_hash}")
        return cached_transcription.read_text(encoding="utf-8")

    # Transcribe using whispercpp server
    logger.info(f"Transcribing audio file: {audio_path}")
    start_time = time.time()

    try:

        texts = []
        #ends = []
        segments, info = WHISPER_ENGINE.transcribe(audio_path, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
          print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
          texts.append(segment.text.strip())
          #ends.append(segment.end)
        transcription = ' '.join(texts).strip()
        #print(f"segments: {ends[-1]}")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"Transcription completed in {elapsed_ms:.2f}ms: '{transcription}'")

        # Save to cache
        if transcription:
            save_transcription_to_cache(audio_hash, transcription)
            logger.info(f"Saved transcription to cache: {audio_hash}")

        return transcription

    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        # Fallback to a generic message if transcription fails
        return "This is the voice you should use for the generation."
