
import functools
from pathlib import Path

from imohash import hashfile

CACHE_BASE = Path.cwd()
CACHE_DIR = CACHE_BASE.joinpath("cache")

@functools.cache
def get_cache_dir() -> Path:
    """Get or create the embedding cache directory"""
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_file_hash(file_path: str) -> str:
    return hashfile(file_path, hexdigest=True)

def get_transcription_filepath_by_hash(filename, hash: str) :
    """Get cached transcription if it exists."""
    cache_file = Path(get_cache_dir()).joinpath(f"{filename}_{hash}_transcription.txt")
    if cache_file.exists():
        return cache_file.resolve(strict=True)
    return None

def save_transcription_to_cache(filename, audio_hash: str, transcription: str) -> None:
    """Save transcription to cache."""
    cache_file = get_cache_dir().joinpath(f"{filename}_{audio_hash}_transcription.txt")
    cache_file.write_text(data=transcription, encoding="utf-8")
