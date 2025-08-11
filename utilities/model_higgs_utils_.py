from typing import Optional, Tuple

import torch
import numpy as np
from faster_whisper import WhisperModel

from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIGGS_ENGINE: Optional[HiggsAudioServeEngine] = None

def initialize_higgs_model(whisper_model: WhisperModel, quantization: bool = False) -> HiggsAudioServeEngine:
    global HIGGS_ENGINE

    if HIGGS_ENGINE is None:
        HIGGS_ENGINE = HiggsAudioServeEngine(
            MODEL_PATH,
            AUDIO_TOKENIZER_PATH,
            device=DEVICE,
            quantization=quantization,
            whisper_model=whisper_model
        )

    return HIGGS_ENGINE

def create_voice_cloning_chatmlsample(ref_audio_path: str, ref_text: str, target_text: str, ref_audio_hash: str = None, ref_config: str = None, raw_audio: Optional[Tuple[np.ndarray, int]] = None, ref_audio_uuid: int = None) -> ChatMLSample:
    """Create ChatMLSample for voice cloning using reference audio.

    Caching is handled automatically by the serve engine during processing.

    Args:
        ref_audio_path: Path to the reference audio file
        ref_text: Transcribed text from the reference audio
        target_text: Text to generate in the reference voice
        ref_audio_hash: Pre-computed hash of the reference audio file (optional)
    """
    audio_content = AudioContent(audio_url=ref_audio_path)
    if ref_audio_hash:
        audio_content._temp_hash = ref_audio_hash          
    
    if raw_audio is not None:
        audio_content.raw_audio = raw_audio

    if ref_audio_uuid is not None:
        audio_content._temp_uuid = ref_audio_uuid

    messages = [
        Message(role="user", content=ref_text),
        Message(role="assistant", content=audio_content),
        Message(role="user", content=target_text),
    ]
    
    return ChatMLSample(messages=messages)


def create_text_to_speech_chatmlsample(text: str) -> ChatMLSample:
    """Create ChatMLSample for basic text-to-speech without voice cloning."""
    messages = [
        Message(role="system", content="Generate audio following instruction."),
        Message(role="user", content=text),
    ]
    return ChatMLSample(messages=messages)
