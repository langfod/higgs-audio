"""Higgs TTS Gradio Wrapper for Zonos Client Compatibility."""

import os
import tempfile
import logging
import torch
import gradio as gr
import soundfile as sf
import langid
import jieba
import re
import copy
import torchaudio
import tqdm
import yaml
import time

from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from typing import List
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import StaticCache
from typing import Optional
from dataclasses import asdict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

BNB_CONF = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


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


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device_id=None,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes,
        use_static_kv_cache=False,
    ):
        if device_id is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = f"cuda:{device_id}"
        self._audio_tokenizer = (
            load_higgs_audio_tokenizer(audio_tokenizer, device=self._device)
            if isinstance(audio_tokenizer, str)
            else audio_tokenizer
        )
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            quantization_config=BNB_CONF,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        *args,
        **kwargs,
    ):
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            
            # Generate audio
            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )


            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


def prepare_generation_context_with_reference_audio(ref_audio_path, audio_tokenizer, scene_prompt=None):
    """Prepare the context for generation using a reference audio file path."""
    system_message = None
    messages = []
    audio_ids = []
    
    if ref_audio_path is not None:
        # Encode the reference audio
        audio_tokens = audio_tokenizer.encode(ref_audio_path)
        audio_ids.append(audio_tokens)
        
        # Create system message without audio prompt in system message
        if scene_prompt:
            system_message = Message(
                role="system",
                content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
            )
        else:
            system_message = Message(
                role="system",
                content="Generate audio following instruction.",
            )
        
        # Add user/assistant pair for voice cloning
        ref_text = "This is the voice you should use for the generation."
        messages.append(
            Message(
                role="user",
                content=ref_text,
            )
        )
        messages.append(
            Message(
                role="assistant",
                content=AudioContent(
                    audio_url=ref_audio_path,
                ),
            )
        )
    else:
        # No reference audio, use default system message
        system_message_l = ["Generate audio following instruction."]
        if scene_prompt:
            system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
        system_message = Message(
            role="system",
            content="\n\n".join(system_message_l),
        )
    
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


# Global variables for model initialization
AUDIO_TOKENIZER = None
MODEL_CLIENT = None

def initialize_higgs_model():
    """Initialize the Higgs model and tokenizer globally."""
    global AUDIO_TOKENIZER, MODEL_CLIENT
    
    logger.info("Loading Higgs audio tokenizer...")
    
    # Set device
    device_id = None
    if torch.cuda.is_available():
        device_id = 0
        device = "cuda:0"
    else:
        device_id = None
        device = "cpu"
    
    # Load audio tokenizer
    AUDIO_TOKENIZER = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=device)
    
    logger.info("Loading Higgs model...")
    
    # Initialize model client
    MODEL_CLIENT = HiggsAudioModelClient(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer=AUDIO_TOKENIZER,
        device_id=device_id,
        max_new_tokens=2048,
        use_static_kv_cache=True,
    )
    
    logger.info("Higgs model successfully loaded and ready for inference!")


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
    dnsmos_overall_slider,  # Ignored parameter
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
    global MODEL_CLIENT, AUDIO_TOKENIZER
    
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
    
    # Prepare generation context
    messages, audio_ids = prepare_generation_context_with_reference_audio(
        ref_audio_path=ref_audio_path,
        audio_tokenizer=AUDIO_TOKENIZER,
        scene_prompt=None,
    )
    
    # Prepare text chunks
    chunked_text = prepare_chunk_text(
        processed_text,
        chunk_method=None,  # No chunking for simplicity
        chunk_max_word_num=200,
        chunk_max_num_turns=1,
    )
    
    logger.info("Chunks used for generation:")
    for idx, chunk_text in enumerate(chunked_text):
        logger.info(f"Chunk {idx}: {chunk_text}")
    
    # Generate audio using Higgs model
    try:
        start = time.time()
        concat_wv, sr, text_output = MODEL_CLIENT.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=None,
            temperature=1.0,
            top_k=50,
            top_p=top_p if top_p > 0 else 0.95,
            ras_win_len=7,
            ras_win_max_num_repeat=2,
            seed=int(seed) if seed else 123,
        )
        end = time.time()
        elapsed_ms = (end - start) * 1000
        logger.info(f"Audio generation completed in {elapsed_ms:.2f} ms")
        
        # Create temporary WAV file - CRITICAL: use delete=False
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write audio to temporary file
        sf.write(temp_path, concat_wv, sr)
        logger.info(f"Audio saved to temporary file: {temp_path}")
        
        # Return the file path (not the audio data) for Gradio/Zonos compatibility
        return temp_path
        
    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}")
        return None


# Initialize the model when the module is loaded
logger.info("Initializing Higgs TTS model...")
initialize_higgs_model()


# Define Gradio interface inputs to match Zonos API exactly
api_inputs = [
    gr.Textbox(label="Model"),  # model
    gr.Textbox(label="Text"),  # text
    gr.Textbox(label="Language"),  # language
    gr.File(label="Speaker Audio"),  # speaker_audio
    gr.File(label="Prefix Audio"),  # prefix_audio
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Happiness"),  # response_tone_happiness
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Sadness"),  # response_tone_sadness
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Disgust"),  # response_tone_disgust
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Fear"),  # response_tone_fear
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Surprise"),  # response_tone_surprise
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Anger"),  # response_tone_anger
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Other"),  # response_tone_other
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Neutral"),  # response_tone_neutral
    gr.Slider(minimum=0.5, maximum=0.8, value=0.7, label="VQ Score"),  # vq_score
    gr.Slider(minimum=22050, maximum=24000, value=24000, label="Fmax (Hz)"),  # fmax
    gr.Slider(minimum=20, maximum=150, value=45, label="Pitch Std"),  # pitch_std
    gr.Slider(minimum=0, maximum=45, value=14.6, label="Speaking Rate"),  # speaking_rate
    gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall"),  # dnsmos_overall_slider
    gr.Checkbox(label="Denoise Speaker"),  # denoise_speaker
    gr.Slider(minimum=1, maximum=10, value=3, label="CFG Scale"),  # cfg_scale
    gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P"),  # top_p
    gr.Slider(minimum=1, maximum=100, value=5, label="Min K"),  # min_k
    gr.Slider(minimum=0.01, maximum=1.0, value=0.05, label="Min P"),  # min_p
    gr.Slider(minimum=0, maximum=1, value=0.5, label="Linear"),  # linear
    gr.Slider(minimum=0, maximum=1, value=0.7, label="Confidence"),  # confidence
    gr.Slider(minimum=0, maximum=1, value=0.3, label="Quadratic"),  # quadratic
    gr.Number(value=123, label="Seed"),  # seed
    gr.Checkbox(label="Randomize Seed"),  # randomize_seed
    gr.Textbox(label="Unconditional Keys"),  # unconditional_keys
]

# Create Gradio interface with queue enabled (CRITICAL for Zonos compatibility)
app = gr.Interface(
    fn=generate_audio,
    inputs=api_inputs,
    outputs=gr.Audio(label="Generated Audio"),
    title="Higgs TTS Zonos-Compatible Wrapper",
    description="Higgs TTS model wrapper that provides Zonos TTS API compatibility for seamless client integration.",
).queue()

if __name__ == "__main__":
    # Launch the server
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )