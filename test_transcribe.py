import sys
import traceback
import warnings
from pathlib import Path

from loguru import logger

from boson_multimodal.text_processing.transcript import Transcript
from utilities.file_utils import get_file_hash
from utilities.model_higgs_utils_ import (create_text_to_speech_chatmlsample)
from utilities.model_whisper_utils import (initialize_whisper_model, transcribe_audio_with_whisper)
from utilities.text_utils import pre_process_text
from utilities.token_cache import enable_token_cache
from utilities.voice_style_describer import initialize_describer_models

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

current_dir = Path.cwd()
sys.path.append(str(current_dir.resolve()))  
       
enable_token_cache(memory_cache=True, disk_cache=True)
try:
    WHISPER_ENGINE = initialize_whisper_model()
    initialize_describer_models(use_accent=True)
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    exit(1)


def generate_audio(

        text,  # Main text input

        speaker_audio = None,  # Reference audio file (Gradio file object)

        speaking_rate  = None,  # Ignored parameter
        dnsmos_overall  = None,  # Ignored parameter

        top_p =0.95,  # Can be used for generation
        top_k = 50,  # Ignored parameter

        seed = 0,  # Can be used for generation
        profiling = False,  # Enable PyTorch profiling
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

        else:
            # Regular text-to-speech mode
            logger.info("Creating text-to-speech sample (no voice cloning)")
            chat_sample = create_text_to_speech_chatmlsample(processed_text)

        return reference_text

    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        print(traceback.format_exc())
        return None


## Test junk
logger.info("Running generate_audio function...")
#test_asset=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
test_asset=Path.cwd().joinpath("assets", "fishaudio_horror.wav")



try:
    text = generate_audio(text="Testing Text.",speaker_audio=test_asset,seed=42, profiling=False)
    print(f"Generated audio with text:\n{text}")
except Exception as e:
    print(traceback.format_exc())
