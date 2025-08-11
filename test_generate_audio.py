import sys
import time
import traceback
from pathlib import Path

import torch
from loguru import logger
import warnings
from utilities.file_utils import get_file_hash
from utilities.model_higgs_utils_ import (create_text_to_speech_chatmlsample,
                                           create_voice_cloning_chatmlsample,
                                           initialize_higgs_model)
from utilities.model_whisper_utils import (initialize_whisper_model,
                                            transcribe_audio_with_whisper)
from utilities.text_utils import pre_process_text
from utilities.token_cache import enable_token_cache, get_cache_stats
from utilities.voice_style_describer import initialize_describer_models
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

current_dir = Path.cwd()
sys.path.append(str(current_dir.resolve()))  
       
enable_token_cache(memory_cache=True, disk_cache=True)
try:
    WHISPER_ENGINE = initialize_whisper_model()
    HIGGS_ENGINE = initialize_higgs_model(whisper_model=WHISPER_ENGINE, quantization=True)
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

        if profiling:
            # Single-run profiling (Strategy A)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_logs"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
            ) as prof:
                torch.cuda.synchronize()  # ensure prior work done
                with torch.inference_mode():
                    torch.cuda.nvtx.range_push("infer_single")
                    higgs_wav_out, higgs_sr = HIGGS_ENGINE.generate_audio_only(
                        chat_ml_sample=chat_sample,
                        max_new_tokens=1000,
                        temperature=0.3,
                        top_k=top_k,
                        top_p=top_p if top_p > 0 else 0.95,
                        ras_win_len=speaking_rate,
                        ras_win_max_num_repeat=int(dnsmos_overall) if dnsmos_overall is not None else 2, 
                        seed=int(seed) if seed else None,
                        stop_strings=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                    )
                    torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()  # flush kernels before exiting profiler
                # Optional: export a chrome trace file (uncomment if desired)
                # prof.export_chrome_trace("profile_logs/run_single_trace.json")
        else:
            higgs_wav_out, higgs_sr = HIGGS_ENGINE.generate_audio_only(
                        chat_ml_sample=chat_sample,
                        max_new_tokens=1000,
                        temperature=0.3,
                        top_k=top_k,
                        top_p=top_p if top_p > 0 else 0.95,
                        ras_win_len=speaking_rate,  # 7,
                        ras_win_max_num_repeat=int(dnsmos_overall) if dnsmos_overall is not None else 2,  # 2,
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


## Test junk
logger.info("Running generate_audio function...")
test_asset=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
try:
    # Run twice to warm model and caches
    [sampling_rate, wav_numpy], seed_int = generate_audio(text="Testing Text.",speaker_audio=test_asset,seed=42, profiling=False)
    #[sampling_rate, wav_numpy], seed_int = generate_audio(text="Testing Text.",speaker_audio=test_asset,seed=42, profiling=True)

except Exception as e:
    print(traceback.format_exc())

sampling_rate, wav_numpy, seed_int = None,  None , 0# Reset for next run
print("Sleeping for 5 seconds to allow model warmup...")
time.sleep(5)

try:
    [sampling_rate, wav_numpy], seed_int = generate_audio(text="Testing Text.",speaker_audio=test_asset,seed=42, profiling=False)
    print(f"Generated audio with sampling rate: {sampling_rate}, seed: {seed_int}")
except Exception as e:
    print(traceback.format_exc())
