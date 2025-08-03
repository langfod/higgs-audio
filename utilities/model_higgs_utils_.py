from typing import Optional, List

import numpy as np
import torch
from loguru import logger

from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse, AsyncHiggsAudioStreamer

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIGGS_ENGINE: Optional[HiggsAudioServeEngine] =None

def initialize_higgs_model(quantization: bool = False) -> HiggsAudioServeEngine:
    global HIGGS_ENGINE

    if HIGGS_ENGINE is None:
        logger.info("Loading Higgs serve engine...")
        try:
            # Initialize serve engine
            HIGGS_ENGINE = HiggsAudioServeEngine(
                MODEL_PATH,
                AUDIO_TOKENIZER_PATH,
                device=DEVICE,
                quantization=quantization
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
    return HIGGS_ENGINE


def create_voice_cloning_chatmlsample(ref_audio_path: str, ref_text: str, target_text: str) -> ChatMLSample:
    """Create ChatMLSample for voice cloning using reference audio."""

    messages = [
        Message(role="user", content=ref_text),
        Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
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

def higgs_generate(
    chat_ml_sample: ChatMLSample,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: float = 0.95,
    stop_strings: Optional[List[str]] = None,
    force_audio_gen: bool = False,
    ras_win_len: Optional[int] = 7,
    ras_win_max_num_repeat: int = 2,
    seed: Optional[int] = None,
):
    """
    Generate audio from a chatml sample.
    Args:
        chat_ml_sample: A chatml sample.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for the generation.
        top_p: The top p to use for the generation.
        stop_strings: A list of strings to stop the generation.
        force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
        ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
        ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
    Returns:
        A dictionary with the following keys:
            audio: The generated audio.
            sampling_rate: The sampling rate of the generated audio.
    """
    # Default stop strings
    if stop_strings is None:
        stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
    if ras_win_len is not None and ras_win_len <= 0:
        ras_win_len = None
    with torch.no_grad():
        inputs = HIGGS_ENGINE._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
        HIGGS_ENGINE._prepare_kv_caches()
        outputs = HIGGS_ENGINE.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stop_strings=stop_strings,
            tokenizer=HIGGS_ENGINE.tokenizer,
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            past_key_values_buckets=HIGGS_ENGINE.kv_caches,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed,
        )
        if len(outputs[1]) > 0:
            wv_list = []
            for output_audio in outputs[1]:
                vq_code = revert_delay_pattern(output_audio).clip(0, HIGGS_ENGINE.audio_codebook_size - 1)[:, 1:-1]
                wv_numpy = HIGGS_ENGINE.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                wv_list.append(wv_numpy)
            wv_numpy = np.concatenate(wv_list)
        else:
            wv_numpy = None
        sampling_rate = HIGGS_ENGINE.audio_tokenizer.sampling_rate
    return wv_numpy, sampling_rate

