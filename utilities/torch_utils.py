
import numpy as np
from typing import Optional, Tuple, Any
import warnings
import torchaudio

# Suppress the specific UserWarning related to torchaudio.load
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec`")

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio_tensor(path: str, device: Optional[str] = None) -> Tuple[Any, int]:
    """Load audio from file into a mono float32 torch tensor on the desired device.

    - Does not resample or trim.
    - Returns (waveform[1, T], sample_rate).
    - Prefers torchcodec when available; falls back to torchaudio.load.
    """
    if torchaudio is None or torch is None:
        raise RuntimeError("PyTorch and torchaudio are required for audio loading")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    wav: Any
    sr: int

    # Try torchcodec first to avoid upcoming torchaudio.load internal changes/warnings
    try:
        from torchcodec.decoders import AudioDecoder  # type: ignore

        dec = AudioDecoder(path)  # keep original source rate/channels
        samples = dec.get_all_samples()
        data = samples.data  # Tensor [C, T] or [T]
        sr = int(samples.sample_rate)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        if data.size(0) > 1:
            data = data.mean(dim=0, keepdim=True)
        wav = data.to(torch.float32)
    except Exception:
        # Fallback to torchaudio
        wav, sr = torchaudio.load(path)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(torch.float32)

    return wav.to(dev, non_blocking=True), sr


def resample_audio_tensor(wav: Any, sr: int, target_sr: int) -> Tuple[Any, int]:
    """Resample a mono audio tensor to target_sr if needed and return (wav, sr).

    Attempts to keep the tensor on the same device. If resampling on the current
    device fails, falls back to CPU resampling and moves back to the original device.
    """
    if AF is None or torchaudio is None or torch is None:
        raise RuntimeError("torchaudio is required for resampling")
    if sr == target_sr:
        return wav, sr

    orig_device = wav.device
    try:
        # Try resampling on the current device (may work depending on build)
        out = AF.resample(wav, sr, target_sr)
        return out, target_sr
    except Exception:
        # Fallback: move to CPU, resample, then move back
        wav_cpu = wav.to("cpu")
        out_cpu = AF.resample(wav_cpu, sr, target_sr)
        return out_cpu.to(orig_device, non_blocking=True), target_sr


def trim_audio_tensor(
    wav: Any,
    sr: int,
    top_db: float = 40.0,
    padding_ms: float = 10.0,
) -> Any:
    """Trim leading/trailing silence using a simple amplitude threshold.

    - Keeps tensor on the same device when possible (GPU-friendly ops).
    - Assumes mono shape [1, T] or [T]; returns [1, T_trim].
    - Threshold is max(abs(wav)) * 10^(-top_db/20), clamped to >= 1e-6.
    - Adds optional left/right padding in milliseconds.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for trimming")

    if wav.dim() == 1:
        mono = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.size(0) == 1:
        mono = wav
    else:
        # Force mono just in case
        mono = wav.mean(dim=0, keepdim=True)

    abs_max = torch.max(torch.abs(mono))
    thr = torch.maximum(abs_max * (10.0 ** (-top_db / 20.0)), torch.tensor(1e-6, device=mono.device))

    mask = torch.abs(mono) > thr
    idx = torch.nonzero(mask.squeeze(0), as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return mono  # all silence

    pad = int((padding_ms / 1000.0) * sr)
    start = int(max(0, int(idx[0].item()) - pad))
    end = int(min(mono.size(1), int(idx[-1].item()) + pad))
    return mono[:, start:end]


def audio_tensor_to_numpy(wav: Any, mono: bool = True, copy: bool = False) -> np.ndarray:
    """Convert a torch audio tensor to a float32 numpy array.

    - If mono=True, squeezes channel dim.
    - Always moves to CPU and detaches.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for conversion to numpy")
    x = wav
    if mono and x.dim() == 2 and x.size(0) == 1:
        x = x.squeeze(0)
    return x.detach().to("cpu").numpy().astype(np.float32, copy=copy)


def load_audio(path: str, target_sr: int = 24000) -> Tuple[Any, int]:
    """Legacy helper: load, resample to target_sr, trim, and return numpy.

    This preserves the previous behavior for existing callers.
    """
    # 1) Load as GPU tensor when available
    wav, sr = load_audio_tensor(path, device=DEVICE)
    # 2) Resample if needed
    wav, sr = resample_audio_tensor(wav, sr, target_sr)
    # 3) Trim
    wav = trim_audio_tensor(wav, sr, top_db=40.0, padding_ms=10.0)
    # 4) Convert to numpy mono
    # y = audio_tensor_to_numpy(wav, mono=True, copy=False)
    return wav, sr

def load_audio_to_np(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file, resample, trim, and return as numpy array."""
    wav, sr = load_audio(path, target_sr)
    return audio_tensor_to_numpy(wav, mono=True, copy=False), sr
