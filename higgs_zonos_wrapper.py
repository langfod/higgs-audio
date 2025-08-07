"""Higgs TTS Gradio Wrapper for Zonos Client Compatibility."""
import asyncio
from pathlib import Path
from typing import Optional

import torch
import gradio as gr
import time
import sys

from loguru import logger

from utilities.model_higgs_utils_ import initialize_higgs_model, create_voice_cloning_chatmlsample, create_text_to_speech_chatmlsample
from utilities.token_cache import enable_token_cache, get_cache_stats
from utilities.model_whisper_utils import initialize_whisper_model, transcribe_audio_with_whisper
from utilities.text_utils import pre_process_text
from utilities.file_utils import get_file_hash

current_dir = Path.cwd()
sys.path.append(str(current_dir.resolve()))

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from faster_whisper import WhisperModel

WHISPER_ENGINE: Optional[WhisperModel]= None
HIGGS_ENGINE: Optional[HiggsAudioServeEngine] =None

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


##### Gradio Stuff
def generate_audio(
    model,  # Ignored parameter for Zonos compatibility
    text,  # Main text input
    language = "en",  # Ignored parameter
    speaker_audio = None,  # Reference audio file (Gradio file object)
    prefix_audio = None,  # Ignored parameter
    response_tone_happiness = None,  # Ignored emotion parameter
    response_tone_sadness = None,  # Ignored emotion parameter
    response_tone_disgust = None,  # Ignored emotion parameter
    response_tone_fear = None,  # Ignored emotion parameter
    response_tone_surprise = None,  # Ignored emotion parameter
    response_tone_anger = None,  # Ignored emotion parameter
    response_tone_other = None,  # Ignored emotion parameter
    response_tone_neutral  = None,  # Ignored emotion parameter
    vq_score  = None,  # Ignored parameter
    fmax  = None,  # Ignored parameter
    pitch_std  = None,  # Ignored parameter
    speaking_rate  = None,  # Ignored parameter
    dnsmos_overall  = None,  # Ignored parameter
    denoise_speaker  = None,  # Ignored parameter
    cfg_scale  = None,  # Ignored parameter
    top_p =0.95,  # Can be used for generation
    top_k = 50,  # Ignored parameter
    min_p  = None,  # Ignored parameter
    linear  = None,  # Ignored parameter
    confidence  = None,  # Ignored parameter
    quadratic  = None,  # Ignored parameter
    seed = 0,  # Can be used for generation
    randomize_seed = False,  # Ignored parameter
    unconditional_keys = None,  # Ignored parameter
):
    """Core function for generating audio using Higgs TTS - must match Zonos API exactly."""


    # Input validation
    if not text or not text.strip():
        logger.error("Empty text provided")
        return None
    processed_text = pre_process_text(text)

    # Get reference audio path from Gradio file object
    print(f"Speaker audio: {speaker_audio}")
    ref_audio_path = None
    if speaker_audio is not None and hasattr(speaker_audio, 'name'):
        ref_audio_path = speaker_audio.name
        logger.info(f"Using reference audio: {ref_audio_path}")

    # Create ChatMLSample for generation
    try:
        if ref_audio_path:
            # Voice cloning mode
            logger.info(f"Creating voice cloning sample with reference audio: {ref_audio_path} and text: {processed_text}")
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
        
        #higgs_audio_response: HiggsAudioResponse = HIGGS_ENGINE.generate(
        #    chat_ml_sample=chat_sample,
        #    max_new_tokens=2048,
        #    temperature=0.3,
        #    top_k=top_k,
        #    top_p=top_p if top_p > 0 else 0.95,
        #    ras_win_len=speaking_rate, #7,
        #    ras_win_max_num_repeat=dnsmos_overall, #2,
        #    seed=int(seed) if seed else None,
        #    stop_strings=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        #)

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
        print(f"📊 Cache Statistics:")
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
        import traceback
        print(traceback.format_exc())
        return None

async def initialize():
    enable_token_cache(memory_cache=True, disk_cache=False)
    higgs_engine, whisper_engine = await asyncio.gather(initialize_higgs_model(quantization=True), initialize_whisper_model())
    return higgs_engine, whisper_engine

# Initialize the model when the module is loaded
enable_token_cache(memory_cache=True, disk_cache=True)
logger.info("Initializing Higgs TTS model...")
HIGGS_ENGINE, WHISPER_ENGINE = asyncio.run(initialize())


#HIGGS_ENGINE = initialize_higgs_model(quantization=True)
#WHISPER_ENGINE = initialize_whisper_model()
# Define Gradio interface inputs to match Zonos API exactly (29 parameters)
api_inputs = [
    gr.Textbox(label="Model"),  # 0: model
    gr.Textbox(label="Text", value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Ladle it over some fresh Khajiit meat. Now smell that. Oh boy this is going to be incredible."),  # 1: text
    gr.Textbox(label="Language"),  # 2: language ("en-us")
    gr.File(label="Speaker Audio"),  # 3: speaker_audio (reference audio file)
    gr.File(label="Prefix Audio"),  # 4: prefix_audio (None)
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Happiness"),  # 5: response_tone_happiness
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Sadness"),  # 6: response_tone_sadness
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Disgust"),  # 7: response_tone_disgust
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Fear"),  # 8: response_tone_fear
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Surprise"),  # 9: response_tone_surprise
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Anger"),  # 10: response_tone_anger
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Other"),  # 11: response_tone_other
    gr.Slider(minimum=0, maximum=1, value=0.2, label="Neutral"),  # 12: response_tone_neutral
    gr.Slider(minimum=0.5, maximum=1.0, value=0.7, label="VQ Score"),  # 13: vq_score
    gr.Slider(minimum=20000, maximum=25000, value=24000, label="Fmax (Hz)"),  # 14: fmax
    gr.Slider(minimum=20, maximum=150, value=45, label="Pitch Std"),  # 15: pitch_std
    #gr.Slider(minimum=0, maximum=50, value=14.6, label="Speaking Rate"),  # 16: speaking_rate
    #gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall"),  # 17: dnsmos_overall
    gr.Slider(minimum=0, maximum=10, value=7, label="RAS Window"),  # 16: speaking_rate
    gr.Slider(minimum=1, maximum=10, value=2, label="Max Repeat"),  # 17: dnsmos_overall
    gr.Checkbox(value=True, label="Denoise Speaker"),  # 18: denoise_speaker
    gr.Slider(minimum=1, maximum=10, value=3, label="CFG Scale"),  # 19: cfg_scale
    gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P"),  # 20: top_p
    gr.Slider(minimum=1, maximum=100, value=50, label="Top K"),  # 21: top_k
    gr.Slider(minimum=0.01, maximum=1.0, value=0.2, label="Min P"),  # 22: min_p
    gr.Checkbox(value=False, label="Linear"),  # 23: linear
    gr.Slider(minimum=0, maximum=1, value=0.7, label="Confidence"),  # 24: confidence
    gr.Checkbox(value=False, label="Quadratic"),  # 25: quadratic
    gr.Number(value=123, label="Seed"),  # 26: seed
    gr.Checkbox(value=False, label="Randomize Seed"),  # 27: randomize_seed
    gr.Textbox(value="[]", label="Unconditional Keys"),  # 28: unconditional_keys (empty list)
]

# Create Gradio interface with queue enabled (CRITICAL for Zonos compatibility)
app = gr.Interface(
    fn=generate_audio,
    inputs=api_inputs,
    outputs=[gr.Audio(label="Generated Audio"), gr.Number(label="Seed")],
    title="Higgs TTS Zonos-Compatible Wrapper",
    description="Higgs TTS model wrapper that provides Zonos TTS API compatibility for seamless client integration.",
    api_name="generate_audio"  # Explicitly set API name for Zonos compatibility
).queue()

if __name__ == "__main__":
    # Launch the server
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )