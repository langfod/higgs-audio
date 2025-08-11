import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from faster_whisper import WhisperModel
from loguru import logger

from utilities.torch_utils import (audio_tensor_to_numpy, load_audio_to_np, resample_audio_tensor)

from .file_utils import (get_cached_transcription, save_transcription_to_cache)
from .voice_style_describer import get_speaker_style_line

WHISPER_MODEL = "sorendal/skyrim-whisper-base-int8"
#WHISPER_MODEL = "Numbat/faster-skyrim-whisper-base.en"
#WHISPER_MODEL = "distil-whisper/distil-large-v3.5-ct2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_ENGINE: Optional[WhisperModel]= None

def initialize_whisper_model():
    global  WHISPER_ENGINE
    
    if WHISPER_ENGINE is None:
            # Initialize Whisper model
            logger.info("Loading Whisper model...")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="bfloat16")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8" if DEVICE=="cuda" else "int8")
            logger.info("Whisper model loaded successfully.")
    return WHISPER_ENGINE


def transcribe_audio_with_whisper(audio_path: str, audio_file_hash: str ,language: str = "en", ref_audio_torch: Optional[Tuple[torch.Tensor, int]] = None, uuid: int = 0, enable_style: bool = True, enable_disk_cache: bool = True) -> str:
    """Transcribe audio using local whispercpp server with caching."""

    filename = Path(audio_path).stem
    # Check cache first
    if enable_disk_cache:
        try:
            cached_transcription = get_cached_transcription(filename, audio_file_hash, uuid)
            logger.info(f"Using cached transcription for {filename} with {audio_file_hash}")
            return cached_transcription
        except FileNotFoundError:
            logger.info(f"No cached transcription found for {filename} with {audio_file_hash}")

    # Ensure model is initialized
    if WHISPER_ENGINE is None:
        initialize_whisper_model()

    logger.info(f"Transcribing audio file: {audio_path}")
    start_time = time.time()

    if audio_path and not ref_audio_torch:
        print("Loading reference audio...")
        ref_audio_np, ref_audio_np_sr = load_audio_to_np(audio_path, target_sr=16000)
    elif ref_audio_torch:
        resample, resample_sr = resample_audio_tensor(ref_audio_torch[0], ref_audio_torch[1], 16000)
        ref_audio_np = audio_tensor_to_numpy(resample, mono=True, copy=False)

    try:
        texts = []
        segments, info = WHISPER_ENGINE.transcribe(
            audio=ref_audio_np if isinstance(ref_audio_np, np.ndarray) else audio_path,
            beam_size=2,
            vad_filter=True,
            without_timestamps=True,
            multilingual=True,
        )
        for segment in segments:
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            texts.append(segment.text.strip())
        language = info.language if info.language else language
        transcription = ' '.join(texts).strip()

        if enable_style:
            # Build speaker style line and prepend
            style_line = get_speaker_style_line((ref_audio_np,resample_sr), transcript=transcription, language_code=language)
            final_text = f"{style_line}\n\n{transcription}"
        else:
            final_text = transcription

        logger.debug(final_text)

        if final_text:
            if enable_disk_cache:
                _save_to_disk(audio_file_hash, filename, uuid, final_text)
                logger.info(f"Saved transcription to cache: {audio_file_hash}")

        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"Transcription completed in {elapsed_ms:.2f}ms")
        return final_text

    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Failed to transcribe audio: {str(e)}")
        # Fallback to a generic message if transcription fails
        return "This is the voice you should use for the generation."

def _save_to_disk(audio_file_hash, filename, uuid, final_text):
    threading.Thread(
                    target=save_transcription_to_cache,
                    args=(filename, audio_file_hash, uuid, final_text),
                    daemon=True
                ).start()