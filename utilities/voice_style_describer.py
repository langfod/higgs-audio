import traceback

from loguru import logger
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Optional dependencies
try:
    import torchaudio
    import torchaudio.functional as AF  # type: ignore
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore
    AF = None  # type: ignore
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import parselmouth
except Exception:  # pragma: no cover
    parselmouth = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
    try:
        # Optional helpers for robust processor/loading and config introspection
        from transformers import AutoFeatureExtractor, AutoProcessor, AutoConfig  # type: ignore
    except Exception:  # pragma: no cover
        AutoFeatureExtractor = None  # type: ignore
        AutoProcessor = None  # type: ignore
        AutoConfig = None  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore
    AutoFeatureExtractor = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoConfig = None  # type: ignore


_EMOTION_PIPE = None
_ACCENT_PIPE = None  # optional accent/dialect classifier (English-focused)

# Model identifiers 
#DEFAULT_SER_MODEL_ID =  "DunnBC22/wav2vec2-base-Speech_Emotion_Recognition"
#"prithivMLmods/Speech-Emotion-Classification"
DEFAULT_SER_MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
#"superb/wav2vec2-large-superb-er" #"superb/hubert-large-superb-er"


DEFAULT_ACCENT_MODEL_ID = "HamzaSidhu786/speech-accent-detection"


@dataclass
class VoiceStyle:
    sentence: str
    pitch_hz: Optional[float] = None
    pitch_sd_hz: Optional[float] = None
    snr_db: Optional[float] = None
    hnr_db: Optional[float] = None
    rate_wps: Optional[float] = None


def _device_index() -> int:
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return 0
    return -1


def _load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 at target_sr, preferring torchcodec when available.

    Falls back to torchaudio.load + AF.resample to maintain current behavior.
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for voice style description")

    # Try torchcodec first to avoid upcoming torchaudio.load internal changes/warnings
    sr = target_sr
    try:
        from torchcodec.decoders import AudioDecoder  # type: ignore

        dec = AudioDecoder(path, sample_rate=target_sr, num_channels=1)
        samples = dec.get_all_samples()
        data = samples.data  # Tensor [C, T], float in [-1, 1]
        sr = int(samples.sample_rate)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.size(0) > 1:
            data = data.mean(dim=0, keepdim=True)
        wav = data
    except Exception:
        #print(traceback.format_exc())
        # Fallback to torchaudio
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != target_sr and AF is not None:
            wav = AF.resample(wav, sr, target_sr)
            sr = target_sr

    # Trim leading/trailing silence using simple energy threshold similar to top_db=40
    y = wav.squeeze(0).numpy()
    thr = float(np.max(np.abs(y)) * (10 ** (-40 / 20)))  # -40 dB of max
    idx = np.where(np.abs(y) > max(thr, 1e-6))[0]
    if idx.size > 0:
        start = int(max(0, idx[0] - int(0.01 * sr)))  # 10ms cushion
        end = int(min(len(y), idx[-1] + int(0.01 * sr)))
        y = y[start:end]
    return y.astype(np.float32, copy=False), sr


def _snr_proxy(y: np.ndarray) -> float:
    p05 = float(np.percentile(np.abs(y), 5))
    p95 = float(np.percentile(np.abs(y), 95))
    snr = 20.0 * np.log10((p95 + 1e-8) / (p05 + 1e-8))
    return float(max(-10.0, min(60.0, snr)))


def _prosody_parselmouth(audio: np.ndarray, sr: int) -> Tuple[float, float, Optional[float]]:
    if parselmouth is None:
        return 0.0, 0.0, None
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=450)
    mean_f0 = float(parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz") or 0.0)
    stdev_f0 = float(parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz") or 0.0)
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
    try:
        mean_hnr = float(parselmouth.praat.call(harm, "Get mean", 0, 0))
    except Exception:
        mean_hnr = None
    return mean_f0, stdev_f0, mean_hnr


def _prosody_fallback(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Fallback F0 using torchaudio's YIN-based detector if Praat is unavailable."""
    if torchaudio is None or AF is None or not TORCH_AVAILABLE:
        return 0.0, 0.0
    try:
        wav = torch.from_numpy(y).float().unsqueeze(0)  # [1, T]
        # YIN detection (returns [n_channels, n_frames])
        f0 = AF.detect_pitch_frequency(
            wav,
            sample_rate=sr,
            frame_time=0.01,
            freq_low=60.0,
            freq_high=450.0,
        )
        f0_vals = f0.squeeze(0).numpy()
        f0_vals = f0_vals[np.isfinite(f0_vals)]
        f0_vals = f0_vals[f0_vals > 0]
        if f0_vals.size == 0:
            return 0.0, 0.0
        return float(np.mean(f0_vals)), float(np.std(f0_vals))
    except Exception:
        return 0.0, 0.0


def _bucket_pitch(f0: float) -> str:
    bucket_pitch = "high"
    if f0 <= 0:
        bucket_pitch = "unknown pitch"
    elif f0 < 120:
        bucket_pitch = "deep"
    elif f0 < 180:
        bucket_pitch = "moderate-low"
    elif f0 < 250:
        bucket_pitch = "moderate"
    logger.debug(f"Detected pitch: {bucket_pitch} using F0 as {f0}")
    return bucket_pitch


def _gender_from_f0(f0: float) -> Tuple[str, float]:
    """Map F0 to an apparent gender label with a confidence score.

    Bands (Hz):
      - <150: apparent male
      - 150–165: apparent male (moderate)
      - 165–180: androgynous/ambiguous
      - 180–195: apparent female (moderate)
      - >195: apparent female

    Confidence:
      - Inside overlap (165–180), highest at the center (≈172.5 Hz), falling to 0 at edges.
      - Outside overlap, grows with distance from the overlap band (normalized by 30 Hz).
    """
    if f0 <= 0:
        logger.debug("Detected gender: unspecified using invalid/nonpositive F0")
        return "unspecified", 0.0

    overlap_low = 165.0
    overlap_high = 180.0
    male_strong = 150.0
    female_strong = 195.0
    center = (overlap_low + overlap_high) / 2.0  # 172.5

    # Label selection
    if f0 < male_strong:
        gender = "male"
    elif f0 < overlap_low:
        gender = "male"  # moderate
    elif f0 <= overlap_high:
        gender = "androgynous"
    elif f0 <= female_strong:
        gender = "female"  # moderate
    else:
        gender = "female"

    # Confidence calculation
    if overlap_low <= f0 <= overlap_high:
        # Peak confidence at the center, 0 at edges
        half_bw = (overlap_high - overlap_low) / 2.0  # 7.5 Hz
        conf = 1.0 - min(1.0, abs(f0 - center) / max(half_bw, 1e-6))
    else:
        # Distance to nearest edge of overlap, normalized by 30 Hz
        dist = (overlap_low - f0) if f0 < overlap_low else (f0 - overlap_high)
        conf = min(1.0, max(0.0, dist / 30.0))

    logger.debug(f"Detected gender: {gender} using F0={f0:.1f} Hz with confidence={conf:.2f}")
    return gender, float(conf)


def _timbre_from_hnr(hnr: Optional[float]) -> Optional[str]:
    timbre = "clear"
    if hnr is None:
        timbre = None
    elif hnr < 7:
        timbre = "harsh/rough"
    elif hnr < 15:
        timbre = "slightly breathy"
    logger.debug(f"Detected timbre: {timbre} using HNR as {hnr}")
    return timbre


def _clarity_from_snr(snr_db: float) -> str:
    clarity = "unclear audio"
    if snr_db > 25:
        clarity = "very clear audio"
    elif snr_db > 15:
        clarity =   "clear audio"
    elif snr_db > 8:
        clarity = "some background noise"
    logger.debug(f"Detected clarity: {clarity} using SNR as {snr_db}")
    return clarity


def _rate_from_transcript(transcript: Optional[str], duration_s: float) -> Optional[float]:
    if not transcript or duration_s <= 0:
        return None
    words = max(1, len(transcript.strip().split()))
    return float(words) / float(duration_s)


def _rate_desc(wps: Optional[float]) -> Optional[str]:
    rate_desc = "fast speaking rate"
    if wps is None:
        rate_desc = None
    elif wps < 2.0:
        rate_desc = "slow speaking rate"
    elif wps <= 3.5:
        rate_desc = "modern speaking rate"
    logger.debug(f"Detected speaking rate: {rate_desc} using WPS as {wps}")
    return rate_desc


def _ensure_emotion_pipe():
    global _EMOTION_PIPE
    """Lazily initialize an accent classifier for supported languages.

    Currently supports English accents only. This function is resilient:
    on any failure it returns None and the describer proceeds without accent.
    """
    if pipeline is None:
        return None
    if _EMOTION_PIPE is not None:
        return _EMOTION_PIPE
    try:
        # Default to a transformers-compatible English accent classifier
        model_id = DEFAULT_SER_MODEL_ID

        feat = None
        # Prefer AutoProcessor if available
        if AutoProcessor is not None:
            try:
                feat = AutoProcessor.from_pretrained(model_id)  # type: ignore
            except Exception:
                feat = None
        if feat is None and AutoFeatureExtractor is not None:
            try:
                feat = AutoFeatureExtractor.from_pretrained(model_id)  # type: ignore
            except Exception:
                feat = None

        if feat is not None:
            _EMOTION_PIPE = pipeline(
                "audio-classification",
                model=model_id,
                feature_extractor=feat,
                device=_device_index(),
            )
        else:
            _EMOTION_PIPE = pipeline(
                "audio-classification",
                model=model_id,
                device=_device_index(),
            )
    except Exception as e:
        logger.error(f"Error during accent pipeline initialization for model '{model_id}': {str(e)}")
        print(traceback.format_exc())
        _EMOTION_PIPE = None
    return _EMOTION_PIPE


# Language-ID pipeline removed: we now rely on caller-provided language_code


def _ensure_accent_pipe(language_code: Optional[str]):
    """Lazily initialize an accent classifier for supported languages.

    Currently supports English accents only. This function is resilient:
    on any failure it returns None and the describer proceeds without accent.
    """
    global _ACCENT_PIPE
    if pipeline is None:
        return None
    # Only initialize for English; keep cold start light
    if not language_code or not str(language_code).lower().startswith("en"):
        return None
    if _ACCENT_PIPE is not None:
        return _ACCENT_PIPE
    try:
        # Default to a transformers-compatible English accent classifier
        model_id = DEFAULT_ACCENT_MODEL_ID

        feat = None
        # Prefer AutoProcessor if available
        if AutoProcessor is not None:
            try:
                feat = AutoProcessor.from_pretrained(model_id)  # type: ignore
            except Exception:
                feat = None
        if feat is None and AutoFeatureExtractor is not None:
            try:
                feat = AutoFeatureExtractor.from_pretrained(model_id)  # type: ignore
            except Exception:
                feat = None

        if feat is not None:
            _ACCENT_PIPE = pipeline(
                "audio-classification",
                model=model_id,
                feature_extractor=feat,
                device=_device_index(),
            )
        else:
            _ACCENT_PIPE = pipeline(
                "audio-classification",
                model=model_id,
                device=_device_index(),
            )
    except Exception as e:
        logger.error(f"Error during accent pipeline initialization for model '{model_id}': {str(e)}")
        print(traceback.format_exc())
        _ACCENT_PIPE = None
    return _ACCENT_PIPE


def _map_en_accent_label(label: str) -> tuple[str, Optional[str]]:
    """Map raw model labels to human-friendly English accent names and locale.

    Returns (name, locale) where locale is an optional BCP47-like tag.
    Unknown labels fall back to capitalized form without locale.
    """
 
    l = label.lower()
    accent_map = (label.strip().title(), None)
    # Common variants observed across community models
    if l == "english" or "england" in l:
        accent_map = ("English", "en-GB")
    elif any(k in l for k in ["us", "american", "usa"]):
        accent_map = ("American", "en-US")
    elif any(k in l for k in ["uk", "british", "england", "gb"]):
        accent_map = ("British", "en-GB")
    elif any(k in l for k in ["australia", "australian", "au"]):
        accent_map = ("Australian", "en-AU")
    elif any(k in l for k in ["canada", "canadian"]):
        accent_map = ("Canadian", "en-CA")
    elif any(k in l for k in ["india", "indian", "in"]):
        accent_map = ("Indian", "en-IN")
    elif any(k in l for k in ["ireland", "irish", "ie"]):
        accent_map = ("Irish", "en-IE")
    elif any(k in l for k in ["northernirish", "northern irish", "northern-irish"]):
        accent_map = ("Northern Irish", "en-GB")
    elif any(k in l for k in ["scotland", "scottish", "sct"]):
        accent_map = ("Scottish", "en-GB-scotland")
    elif any(k in l for k in ["wales", "welsh", "cy"]):
        accent_map = ("Welsh", "en-GB-wales")
    elif any(k in l for k in ["newzealand", "new zealand", "nz"]):
        accent_map = ("New Zealand", "en-NZ")
    elif any(k in l for k in ["southafrican", "south african", "za"]):
        accent_map = ("South African", "en-ZA")
    elif any(k in l for k in ["africa", "african"]):
        accent_map = ("African", None)
    elif l == "unknown":
        accent_map = ("Unknown", None)
    logger.debug(f"Mapped accent label '{label}' to {accent_map[0]} with locale {accent_map[1]}")
    # Fallback: title-case the label
    return accent_map


def _tone_from_label(label: str) -> str:
    m = {
        "neutral": "neutral tone",
        "happy": "friendly tone",
        "angry": "tense tone",
        "sad": "subdued tone",
        "fear": "tense tone",
        "disgust": "tense tone",
        "surprise": "energetic tone",
    }
    logger.debug(f"Detected tone: {m.get(label.lower(), 'neutral tone')} using label as {label}")
    return m.get(label.lower(), "neutral tone")


def describe_speaker(
    audio_np: Tuple[np.ndarray, int],
    transcript: Optional[str] = None,
    assume_english: bool = True,
    language_code: Optional[str] = None,
) -> VoiceStyle:
    y, sr = audio_np
    duration = float(len(y)) / float(sr) if sr else 0.0

    # Prosody
    f0, f0_sd, hnr = _prosody_parselmouth(y, sr)
    if f0 <= 0:
        f0, f0_sd = _prosody_fallback(y, sr)

    snr_db = _snr_proxy(y)
    clarity = _clarity_from_snr(snr_db)
    pitch_bucket = _bucket_pitch(f0)
    apparent_gender, _ = _gender_from_f0(f0)
    timbre = _timbre_from_hnr(hnr)

    # Speaking rate
    wps = _rate_from_transcript(transcript, duration)
    rate = _rate_desc(wps)

    # Language handling: rely on provided language_code; optionally default to English
    lang = language_code if language_code else ("en" if assume_english else None)

    # Emotion (optional)
    tone = "neutral tone"
    if _ensure_emotion_pipe() is not None:
        try:
            e1 = _EMOTION_PIPE({"array": y, "sampling_rate": sr}, top_k=1)
            logger.debug(f"Emotion detection results: {e1}")
            e = e1[0].get("label", "neutral")  # type: ignore
            tone = _tone_from_label(e)
        except Exception:
            pass

    # Accent (optional; English-only; confidence/margin gated)
    accent_name = None
    accent_locale = None
    accent_strength = None
    if lang and str(lang).lower().startswith("en"):
        pipe = _ensure_accent_pipe(lang)
        if pipe is not None:
            try:
                results = pipe({"array": y, "sampling_rate": sr}, top_k=3)  # type: ignore
                if isinstance(results, dict):
                    results = [results]
                # Sort by score descending (some pipelines already return sorted)
                results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
                logger.debug(f"Accent detection sorted results: {[f'{result.get('label', '')}: {result.get('score', 0):.6f}' for result in results]}")
                if results:
                    top = results[0]
                    top2 = results[1] if len(results) > 1 else None
                    score = float(top.get("score", 0.0))
                    margin = float(score - float(top2.get("score", 0.0)) if top2 else score)
                    # Gate: require decent absolute confidence and separation
                    if score >= 0.60 and margin >= 0.15:
                        name, loc = _map_en_accent_label(str(top.get("label", "")))
                        accent_name, accent_locale = name, loc
                        # Strength heuristics based on confidence
                        if score >= 0.85:
                            accent_strength = "strong"
                        elif score >= 0.72:
                            accent_strength = "noticeable"
                        else:
                            accent_strength = "slight"
            except Exception as e:
                logger.error(f"Error during accent generation: {str(e)}")
                print(traceback.format_exc())
                pass

    # Simple non-human/robotic hint
    nonhuman_hint = None
    logger.debug(f"Non human=(f0>0 and f0_sd<10 and hnr>25) F0: {f0}, F0 SD: {f0_sd}, HNR: {hnr}")
    if f0 > 0 and f0_sd < 10 and (hnr is not None and hnr > 25):
        nonhuman_hint = "robotic/tts-like"

    parts = []
    if lang and str(lang).lower().startswith("en"):
        # Include regional tag when confident
        if accent_name:
            if accent_name.lower() == "english":
                parts.append("English")
            else:
                parts.append(f"{accent_name} English")
        else:
            parts.append("English")
    parts.append(apparent_gender)
    #parts.append("voice")
    if rate:
        parts.append(rate)
    parts.append(f"{pitch_bucket} pitch")
    if timbre and timbre != "clear":
        parts.append(timbre)
    parts.append(tone)
    parts.append(clarity)
    if accent_name and accent_strength:
        # Show accent detail as a suffix; avoid clutter when not confident
        if accent_locale:
            parts.append(f"accent: {accent_locale} ({accent_strength})")
        else:
            parts.append(f"accent: {accent_name} ({accent_strength})")
    if nonhuman_hint:
        parts.append(nonhuman_hint)

    sentence = "speaker: " + ", ".join(parts) + "."
    return VoiceStyle(
        sentence=sentence,
        pitch_hz=round(float(f0), 1) if f0 else None,
        pitch_sd_hz=round(float(f0_sd), 1) if f0_sd else None,
        snr_db=round(float(snr_db), 1),
        hnr_db=round(float(hnr), 1) if (hnr is not None) else None,
        rate_wps=round(float(wps), 2) if (wps is not None) else None,
    )


def get_speaker_style_line(
    audio_np: Tuple[np.ndarray, int],
    transcript: Optional[str] = None,
    language_code: Optional[str] = None,
) -> str:
    """Return a single line prefixed with '- ' describing the speaker style."""
    try:
        style = describe_speaker(audio_np, transcript=transcript, language_code=language_code)
        return f"- {style.sentence}\n- extra: pitch={style.pitch_hz}Hz pitch_sd={style.pitch_sd_hz}Hz snr={style.snr_db}dB hnr={style.hnr_db}dB rate={style.rate_wps}WPS"
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error during audio generation: {str(e)}")
        # Never block transcription if style estimation fails
        return "- speaker: neutral tone, unknown pitch, audio quality unknown."


def get_accent_model_labels() -> Optional[list[str]]:
    """Return the list of class labels from the configured accent model.

    Tries the loaded pipeline first; if not available, attempts to read the
    model config using AutoConfig. Returns None on failure.
    """
    # Prefer loaded pipeline
    if _ACCENT_PIPE is not None:
        try:
            cfg = getattr(_ACCENT_PIPE.model, "config", None)
            if cfg is not None and hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict):
                # Sort by id key if ids are numeric
                try:
                    return [cfg.id2label[i] for i in sorted(cfg.id2label.keys())]
                except Exception:
                    # Fallback: natural order of dict values
                    return list(cfg.id2label.values())
        except Exception:
            pass

    # Try loading config directly if AutoConfig is available
    try:
        if AutoConfig is None:
            return None
        model_id = DEFAULT_ACCENT_MODEL_ID
        cfg = AutoConfig.from_pretrained(model_id)  # type: ignore
        if hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict):
            try:
                return [cfg.id2label[i] for i in sorted(cfg.id2label.keys())]
            except Exception:
                return list(cfg.id2label.values())
    except Exception as e:
        logger.debug(f"Accent labels could not be loaded: {e}")
    return None


def initialize_describer_models(
    device: Optional[int] = None,
    use_emotion: bool = True,
    use_accent: bool = False,
) -> None:
    """Pre-initialize local pipelines used by the voice style describer.

    - device: CUDA index (0-based) or CPU when None/-1; defaults to auto-detect
    - use_emotion: load speech emotion recognition pipeline

    Safe to call multiple times; only initializes missing components.
    """
    dev = _device_index() if device is None else device

    # Log availability of optional analyzers
    if parselmouth is None:
        logger.warning("parselmouth not available; falling back to YIN pitch detector")
    if torchaudio is None:
        logger.warning("torchaudio not available; audio loading/resample may fail")

    if pipeline is None:
        logger.warning("transformers not available; emotion/accent pipelines cannot be initialized")
        return

    global _EMOTION_PIPE, _ACCENT_PIPE
    if use_emotion and _EMOTION_PIPE is None:
        try:
            logger.info("Initializing emotion pipeline (superb/hubert-large-superb-er)...")
            _EMOTION_PIPE = _ensure_emotion_pipe()
            logger.info("Emotion pipeline initialized")
        except Exception as e:
            logger.warning(f"Emotion pipeline unavailable; continuing without SER: {e}")
            _EMOTION_PIPE = None

    if use_accent and _ACCENT_PIPE is None:
        try:
            # Only initialize for English by default; honor ACCENT_MODEL_ID if set
            logger.info("Initializing accent pipeline (English accent classifier)...")
            _ACCENT_PIPE = _ensure_accent_pipe(language_code="en")
            if _ACCENT_PIPE is not None:
                logger.info("Accent pipeline initialized")
            else:
                logger.warning("Accent pipeline unavailable or not configured; continuing without accent")
        except Exception as e:
            logger.warning(f"Accent pipeline unavailable; continuing without accent: {e}")
            _ACCENT_PIPE = None

