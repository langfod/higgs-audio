"""Discrete audio token caching utilities for Higgs Audio model."""
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from .file_utils import get_cache_dir, get_file_hash

_AUDIO_TOKEN_CACHE = {}
_CACHE_ENABLED = False
_DISK_CACHE_ENABLED = False

def enable_token_cache(memory_cache: bool = True, disk_cache: bool = True):
    """Enable audio token caching.

    Args:
        memory_cache: Enable in-memory caching
        disk_cache: Enable disk-based caching
    """
    global _CACHE_ENABLED, _DISK_CACHE_ENABLED
    _CACHE_ENABLED = memory_cache
    _DISK_CACHE_ENABLED = disk_cache
    logger.info(f"Audio token cache enabled - Memory: {memory_cache}, Disk: {disk_cache}")

def disable_token_cache():
    """Disable audio token caching and clear memory cache."""
    global _CACHE_ENABLED, _DISK_CACHE_ENABLED, _AUDIO_TOKEN_CACHE
    _CACHE_ENABLED = False
    _DISK_CACHE_ENABLED = False
    _AUDIO_TOKEN_CACHE.clear()
    logger.info("Audio token cache disabled and memory cache cleared")

def is_cache_enabled() -> Tuple[bool, bool]:
    """Return current cache status (memory_enabled, disk_enabled)."""
    return _CACHE_ENABLED, _DISK_CACHE_ENABLED

def clear_memory_cache():
    """Clear the in-memory audio token cache."""
    global _AUDIO_TOKEN_CACHE
    _AUDIO_TOKEN_CACHE.clear()
    logger.info("Memory audio token cache cleared")

def get_cache_stats() -> dict:
    """Get current cache statistics."""
    disk_token_count = 0
    if _DISK_CACHE_ENABLED:

       cache_dir = get_cache_dir()
       if cache_dir.exists():
          disk_token_count = len(list(cache_dir.glob("*_tokens.pt")))

    return {
        "memory_cache_enabled": _CACHE_ENABLED,
        "disk_cache_enabled": _DISK_CACHE_ENABLED,
        "memory_token_cache_size": len(_AUDIO_TOKEN_CACHE),
        "disk_token_cache_size": disk_token_count,
        "total_cached_items": len(_AUDIO_TOKEN_CACHE) + disk_token_count,
        "memory_token_cache_keys": list(_AUDIO_TOKEN_CACHE.keys())
    }

def _get_filename_from_path(audio_path: str) -> str:
    """Extract filename from audio path for cache naming."""
    return Path(audio_path).stem

def get_cached_audio_tokens(audio_path: str, device: torch.device, audio_hash: str = None, filename: str = None) -> Optional[torch.Tensor]:
    """Get cached discrete audio tokens if available.

    Args:
        audio_path: Path to the audio file
        device: Target device for the tensors
        audio_hash: Pre-computed hash (optional, will compute if not provided)
        filename: Pre-computed filename (optional, will compute if not provided)

    Returns:
        Cached audio tokens tensor if available, None otherwise
    """
    if not _CACHE_ENABLED and not _DISK_CACHE_ENABLED:
        return None

    if audio_hash is None:
        audio_hash = get_file_hash(audio_path)

    # Try memory cache first
    if _CACHE_ENABLED and audio_hash in _AUDIO_TOKEN_CACHE:
        audio_tokens = _AUDIO_TOKEN_CACHE[audio_hash]
        audio_tokens = audio_tokens.to(device)
        logger.info(f"Using memory cached audio tokens for {audio_path}")
        return audio_tokens

    # Try disk cache
    if _DISK_CACHE_ENABLED:
        audio_tokens = _load_audio_tokens_from_disk(audio_path, device, audio_hash, filename)
        if audio_tokens is not None:
            if _CACHE_ENABLED:
                _AUDIO_TOKEN_CACHE[audio_hash] = audio_tokens.to(device)
            return audio_tokens

    return None

def cache_audio_tokens(audio_path: str, audio_tokens: torch.Tensor, audio_hash: str = None, filename: str = None) -> None:
    """Cache discrete audio tokens.

    Args:
        audio_path: Path to the audio file
        audio_tokens: Audio tokens tensor to cache
        audio_hash: Pre-computed hash (optional, will compute if not provided)
        filename: Pre-computed filename (optional, will compute if not provided)
    """
    if not _CACHE_ENABLED and not _DISK_CACHE_ENABLED:
        return

    if audio_hash is None:
        audio_hash = get_file_hash(audio_path)

    if _CACHE_ENABLED:
        _AUDIO_TOKEN_CACHE[audio_hash] = audio_tokens.clone()
        logger.info(f"Cached audio tokens in memory for {audio_path} (device: {audio_tokens.device})")

    if _DISK_CACHE_ENABLED:
        threading.Thread(
            target=_save_audio_tokens_to_disk,
            args=(audio_path, audio_tokens, audio_hash, filename),
            daemon=True
        ).start()

def _save_audio_tokens_to_disk(audio_path: str, audio_tokens: torch.Tensor, audio_hash: str = None, filename: str = None) -> None:
    """Save discrete audio tokens to disk cache."""
    if not _DISK_CACHE_ENABLED:
        return

    try:
        if audio_hash is None:
            audio_hash = get_file_hash(audio_path)
        if filename is None:
            filename = _get_filename_from_path(audio_path)

        cache_file = get_cache_dir().joinpath(f"{filename}_{audio_hash}_tokens.pt")

        torch.save(audio_tokens.cpu(), cache_file)
        logger.info(f"Saved audio tokens to disk: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save audio tokens to disk for path {audio_path}: {str(e)}")

def _load_audio_tokens_from_disk(audio_path: str, device: torch.device, audio_hash: str = None, filename: str = None) -> Optional[torch.Tensor]:
    """Load discrete audio tokens from disk cache."""
    if not _DISK_CACHE_ENABLED:
        return None

    try:
        if audio_hash is None:
            audio_hash = get_file_hash(audio_path)
        if filename is None:
            filename = _get_filename_from_path(audio_path)

        cache_file = get_cache_dir().joinpath(f"{filename}_{audio_hash}_tokens.pt")
        if not cache_file.exists():
            return None

        audio_tokens = torch.load(cache_file, map_location=device)
        logger.info(f"Loaded audio tokens from disk: {cache_file}")
        return audio_tokens
    except Exception as e:
        logger.warning(f"Failed to load audio tokens from disk for path {audio_path}: {str(e)}")
        return None
