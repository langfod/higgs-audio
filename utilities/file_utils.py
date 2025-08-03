import datetime
import functools
import os

import psutil
from imohash import hashfile
from pathlib import Path

CACHE_BASE = Path.cwd()
CACHE_DIR = CACHE_BASE.joinpath("cache")

@functools.lru_cache(1)
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_embed_cache_dir() -> Path:
    """Get or create the conditionals cache directory"""
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    cache_dir = Path(CACHE_BASE).joinpath(formatted_start_time).joinpath(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_file_hash(file_path: str) -> str:
    return hashfile(file_path, hexdigest=True)

def get_transcription_filepath_by_hash( hash: str) :
    """Get cached transcription if it exists."""
    cache_file = Path(get_embed_cache_dir()).joinpath(f"{hash}_.txt")
    if cache_file.exists():
        return cache_file.resolve(strict=True)
    return None

def get_embed_filepath_by_hash( hash: str) :
    """Get cached transcription if it exists."""
    cache_file = Path(get_embed_cache_dir()).joinpath(f"{hash}.pt")
    if cache_file.exists():
        return cache_file.resolve(strict=True)
    return None

def save_transcription_to_cache(audio_hash: str, transcription: str) -> None:
    """Save transcription to cache."""
    cache_file = get_embed_cache_dir().joinpath(f"{audio_hash}_transcription.txt")
    cache_file.write_text(data=transcription, encoding="utf-8")

@functools.lru_cache(1)
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir
