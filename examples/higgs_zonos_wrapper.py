"""Higgs TTS Gradio Wrapper for Zonos Client Compatibility."""

import os
import tempfile
import base64
import torch
import gradio as gr
import soundfile as sf
import langid
import jieba
import re
import time
import hashlib
import requests
import json

from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTION_CACHE_DIR = os.path.join(CURR_DIR, "transcription_cache")

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
WHISPER_API_URL = "http://localhost:8080/inference"


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        """: '"',  # left double quotation
        """: '"',  # right double quotation
        "'": "'",  # left single quotation
        "'": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode audio file content to base64 format for AudioContent."""
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def get_audio_file_md5(file_path: str) -> str:
    """Get MD5 hash of audio file for caching."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_cached_transcription(audio_md5: str) -> str:
    """Get cached transcription if it exists."""
    cache_file = os.path.join(TRANSCRIPTION_CACHE_DIR, f"{audio_md5}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def save_transcription_to_cache(audio_md5: str, transcription: str) -> None:
    """Save transcription to cache."""
    os.makedirs(TRANSCRIPTION_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(TRANSCRIPTION_CACHE_DIR, f"{audio_md5}.txt")
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(transcription)


def transcribe_audio_with_whisper(audio_path: str) -> str:
    """Transcribe audio using local whispercpp server with caching."""
    # Get MD5 hash for caching
    audio_md5 = get_audio_file_md5(audio_path)
    logger.info(f"Audio file MD5: {audio_md5}")
    
    # Check cache first
    cached_transcription = get_cached_transcription(audio_md5)
    if cached_transcription:
        logger.info(f"Using cached transcription for {audio_md5}")
        return cached_transcription
    
    # Transcribe using whispercpp server
    logger.info(f"Transcribing audio file: {audio_path}")
    start_time = time.time()
    
    try:
        payload = {
            "temperature": 0,
            "response_format": "json"
        }
        
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/wav")
            }
            response = requests.post(WHISPER_API_URL, files=files, data=payload, timeout=60)
            response.raise_for_status()
        
        # Handle JSON response
        if response.headers.get("Content-Type", "").startswith("application/json"):
            transcription = response.json().get("text", "").strip()
        else:
            transcription = response.text.strip()
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"Transcription completed in {elapsed_ms:.2f}ms: '{transcription}'")
        
        # Save to cache
        if transcription:
            save_transcription_to_cache(audio_md5, transcription)
            logger.info(f"Saved transcription to cache: {audio_md5}")
        
        return transcription
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {str(e)}")
        # Fallback to a generic message if transcription fails
        return "This is the voice you should use for the generation."


def create_voice_cloning_sample(ref_audio_path: str, target_text: str) -> ChatMLSample:
    """Create ChatMLSample for voice cloning using reference audio."""
    # Transcribe the reference audio to get the actual spoken text
    reference_text = transcribe_audio_with_whisper(ref_audio_path)
    reference_audio = encode_base64_content_from_file(ref_audio_path)
    
    messages = [
        Message(role="user", content=reference_text),
        Message(role="assistant", content=AudioContent(raw_audio=reference_audio, audio_url="placeholder")),
        Message(role="user", content=target_text),
    ]
    return ChatMLSample(messages=messages)


def create_text_to_speech_sample(text: str) -> ChatMLSample:
    """Create ChatMLSample for basic text-to-speech without voice cloning."""
    messages = [
        Message(role="system", content="Generate audio following instruction."),
        Message(role="user", content=text),
    ]
    return ChatMLSample(messages=messages)


# Global variable for serve engine
SERVE_ENGINE = None

def initialize_higgs_model():
    """Initialize the Higgs serve engine globally."""
    global SERVE_ENGINE
    
    logger.info("Loading Higgs serve engine...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize serve engine
    SERVE_ENGINE = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )
    
    logger.info("Higgs serve engine successfully loaded and ready for inference!")


def generate_audio(
    model,  # Ignored parameter for Zonos compatibility
    text,  # Main text input
    language,  # Ignored parameter
    speaker_audio,  # Reference audio file (Gradio file object)
    prefix_audio,  # Ignored parameter
    response_tone_happiness,  # Ignored emotion parameter
    response_tone_sadness,  # Ignored emotion parameter
    response_tone_disgust,  # Ignored emotion parameter
    response_tone_fear,  # Ignored emotion parameter
    response_tone_surprise,  # Ignored emotion parameter
    response_tone_anger,  # Ignored emotion parameter
    response_tone_other,  # Ignored emotion parameter
    response_tone_neutral,  # Ignored emotion parameter
    vq_score,  # Ignored parameter
    fmax,  # Ignored parameter
    pitch_std,  # Ignored parameter
    speaking_rate,  # Ignored parameter
    dnsmos_overall,  # Ignored parameter
    denoise_speaker,  # Ignored parameter
    cfg_scale,  # Ignored parameter
    top_p,  # Can be used for generation
    min_k,  # Ignored parameter
    min_p,  # Ignored parameter
    linear,  # Ignored parameter
    confidence,  # Ignored parameter
    quadratic,  # Ignored parameter
    seed,  # Can be used for generation
    randomize_seed,  # Ignored parameter
    unconditional_keys,  # Ignored parameter
):
    """Core function for generating audio using Higgs TTS - must match Zonos API exactly."""
    global SERVE_ENGINE
    
    logger.info(f"Processing TTS request for text: {text}")
    
    # Input validation
    if not text or not text.strip():
        logger.error("Empty text provided")
        return None
    
    # Get reference audio path from Gradio file object
    ref_audio_path = None
    if speaker_audio is not None and hasattr(speaker_audio, 'name'):
        ref_audio_path = speaker_audio.name
        logger.info(f"Using reference audio: {ref_audio_path}")
    
    # Text preprocessing - reuse from generation.py
    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    speaker_tags = sorted(set(pattern.findall(text)))
    
    # Perform basic normalization
    processed_text = normalize_chinese_punctuation(text)
    # Other normalizations
    processed_text = processed_text.replace("(", " ")
    processed_text = processed_text.replace(")", " ")
    processed_text = processed_text.replace("°F", " degrees Fahrenheit")
    processed_text = processed_text.replace("°C", " degrees Celsius")
    
    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        processed_text = processed_text.replace(tag, replacement)
    
    lines = processed_text.split("\n")
    processed_text = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    processed_text = processed_text.strip()
    
    if not any([processed_text.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        processed_text += "."
    
    # Create ChatMLSample for generation
    try:
        if ref_audio_path:
            # Voice cloning mode
            logger.info(f"Creating voice cloning sample with reference audio: {ref_audio_path} and text: {processed_text}")
            chat_sample = create_voice_cloning_sample(ref_audio_path, processed_text)
        else:
            # Regular text-to-speech mode
            logger.info("Creating text-to-speech sample (no voice cloning)")
            chat_sample = create_text_to_speech_sample(processed_text)
        
        # Generate audio using serve engine
        logger.info("Starting audio generation...")
        start = time.time()
        
        response: HiggsAudioResponse = SERVE_ENGINE.generate(
            chat_ml_sample=chat_sample,
            max_new_tokens=2048,
            temperature=0.3,
            top_k=50,
            top_p=top_p if top_p > 0 else 0.95,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
            seed=int(seed) if seed else None,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        end = time.time()
        elapsed_ms = (end - start) * 1000
        logger.info(f"Audio generation completed in {elapsed_ms:.2f} ms")
        
        # Create temporary WAV file - CRITICAL: use delete=False
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write audio to temporary file
        sf.write(temp_path, response.audio, response.sampling_rate)
        logger.info(f"Audio saved to temporary file: {temp_path}")
        logger.info(f"Generated text: {response.generated_text}")
        
        # Return the file path (not the audio data) for Gradio/Zonos compatibility
        return temp_path
        
    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        return None


# Initialize the model when the module is loaded
logger.info("Initializing Higgs TTS model...")
initialize_higgs_model()


# Define Gradio interface inputs to match Zonos API exactly (29 parameters)
api_inputs = [
    gr.Textbox(label="Model"),  # 0: model
    gr.Textbox(label="Text"),  # 1: text
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
    gr.Slider(minimum=0, maximum=50, value=14.6, label="Speaking Rate"),  # 16: speaking_rate
    gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall"),  # 17: dnsmos_overall
    gr.Checkbox(value=True, label="Denoise Speaker"),  # 18: denoise_speaker
    gr.Slider(minimum=1, maximum=10, value=3, label="CFG Scale"),  # 19: cfg_scale
    gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P"),  # 20: top_p
    gr.Slider(minimum=1, maximum=100, value=1, label="Min K"),  # 21: min_k
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
    outputs=gr.Audio(label="Generated Audio"),
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