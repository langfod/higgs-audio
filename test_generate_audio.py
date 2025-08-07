import asyncio
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import time
import sys
import traceback
import pandas as pd
from loguru import logger

current_dir = Path.cwd()
sys.path.append(str(current_dir.resolve()))

from utilities.model_higgs_utils_ import initialize_higgs_model, create_voice_cloning_chatmlsample, create_text_to_speech_chatmlsample
from utilities.token_cache import enable_token_cache, get_cache_stats, cache_audio_tokens
from utilities.model_whisper_utils import initialize_whisper_model, transcribe_audio_with_whisper
from utilities.text_utils import pre_process_text
from utilities.file_utils import get_file_hash

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from faster_whisper import WhisperModel

WHISPER_ENGINE: Optional[WhisperModel]= None
HIGGS_ENGINE: Optional[HiggsAudioServeEngine] =None

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_audio(

        text,  # Main text input

        speaker_audio = None,  # Reference audio file (Gradio file object)

        speaking_rate  = None,  # Ignored parameter
        dnsmos_overall  = None,  # Ignored parameter

        top_p =0.95,  # Can be used for generation
        top_k = 50,  # Ignored parameter

        seed = 0,  # Can be used for generation

):
    """Core function for generating audio using Higgs TTS - must match Zonos API exactly."""


    # Input validation
    if not text or not text.strip():
        logger.error("Empty text provided")
        return None
    processed_text = pre_process_text(text)

    # Get reference audio path from Gradio file object
    ref_audio_path = None
    if speaker_audio is not None:
        ref_audio_path = speaker_audio
        logger.info(f"Using reference audio: {ref_audio_path}")

    # Create ChatMLSample for generation
    try:
        if ref_audio_path:
            # Voice cloning mode
            logger.info \
                (f"Creating voice cloning sample with reference audio: {ref_audio_path} and text: {processed_text}")
            ref_audio_hash = get_file_hash(ref_audio_path)
            reference_text = transcribe_audio_with_whisper(ref_audio_path, ref_audio_hash)

            chat_sample = create_voice_cloning_chatmlsample(
                ref_audio_path=ref_audio_path,
                ref_text=reference_text,
                target_text=processed_text,
                ref_audio_hash=ref_audio_hash
            )
        else:
            # Regular text-to-speech mode
            logger.info("Creating text-to-speech sample (no voice cloning)")
            chat_sample = create_text_to_speech_chatmlsample(processed_text)

        # Generate audio using serve engine
        logger.info("Starting audio generation...")
        start = time.time()


        higgs_wav_out, higgs_sr = HIGGS_ENGINE.generate_audio_only(
            chat_ml_sample=chat_sample,
            max_new_tokens=2048,
            temperature=0.3,
            top_k=top_k,
            top_p=top_p if top_p > 0 else 0.95,
            ras_win_len=speaking_rate,  # 7,
            ras_win_max_num_repeat=dnsmos_overall,  # 2,
            seed=int(seed) if seed else None,
            stop_strings=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        )
        end = time.time()
        elapsed_ms = (end - start) * 1000
        logger.info(f"Audio generation completed in {elapsed_ms:.2f} ms")

        stats = get_cache_stats()
        print(f"Cache Statistics:")
        print(f"  Memory cache enabled: {stats['memory_cache_enabled']}")
        print(f"  Disk cache enabled: {stats['disk_cache_enabled']}")
        print(f"  Memory token cache: {stats['memory_token_cache_size']} items")
        print(f"  Disk token cache: {stats['disk_token_cache_size']} files")
        print(f"  Total cached items: {stats['total_cached_items']}")
        if stats['memory_token_cache_keys']:
            print(f"  Token cache keys: {stats['memory_token_cache_keys'][:3]}...")  # Show first 3 keys

        return (higgs_sr, higgs_wav_out), seed

    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        print(traceback.format_exc())
        return None

logger.info("Enabling token cache...")
enable_token_cache(memory_cache=True, disk_cache=False)

async def initialize():
    global HIGGS_ENGINE, WHISPER_ENGINE
    try:
        logger.info("Initializing Higgs TTS model...")
        HIGGS_ENGINE = await initialize_higgs_model(quantization=True)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Failed to load Higgs model: {str(e)}")
        sys.exit(1)
    try:
        logger.info("Initializing Whisper model...")
        WHISPER_ENGINE = await initialize_whisper_model()
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Failed to load Whisper model: {str(e)}")
        sys.exit(1)

asyncio.run(initialize())
if HIGGS_ENGINE is None:
    logger.error("Higgs TTS model initialization failed.")
    sys.exit(1)

if WHISPER_ENGINE is None:
    logger.error("Whisper model initialization failed.")
    sys.exit(1)

logger.info("Running generate_audio function...")
sampling_rate, wav_numpy, seed_int = 0, None, 0
#expected_wav_numpy = np.load('test_generate_audio_output.npy')
try:
    
    [sampling_rate, wav_numpy], seed_int = generate_audio(text="Testing Text.",speaker_audio="assets\\dlc1seranavoice.wav",seed=42)
    print(f"Generated audio with sampling rate: {sampling_rate}, seed: {seed_int}")

    #df_describe = pd.DataFrame(wav_numpy)
    #print(df_describe.describe())
    #if np.array_equal(wav_numpy, expected_wav_numpy):
    #    print("Verification successful: Generated wav matches expected_wav_numpy.")
    #else:
    #    print("Warning: Generated wav does not match expected_wav_numpy.")
    #if sampling_rate != 24000:
    #    print(f"Warning: Expected sampling rate of 24000, but got {sampling_rate}")
    #if seed_int != 42:
    #    print(f"Warning: Expected seed of 42, but got {seed_int}")
except Exception as e:
    print(traceback.format_exc())
    sys.exit(1)
