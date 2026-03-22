"""Unified path management for data and cache directories"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


def get_repo_root() -> Path:
    """Get repository root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_data_root() -> Path:
    """
    Get data root directory.

    Source:
    - Environment variable DATA_ROOT
    - If unset, default to <repo_root>/data
    """
    repo_root = get_repo_root()

    # Check environment variable first
    data_root = os.getenv("DATA_ROOT")
    if data_root:
        return Path(data_root).resolve()

    # Fallback to default data directory under repo root
    return (repo_root / "data").resolve()


def get_cache_root() -> Path:
    """
    Get cache root directory.

    Source:
    - Environment variable CACHE_ROOT
    - If unset, default to <repo_root>/cache
    """
    repo_root = get_repo_root()

    # Check environment variable first
    cache_root = os.getenv("CACHE_ROOT")
    if cache_root:
        path = Path(cache_root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Fallback to default cache directory under repo root
    path = (repo_root / "cache").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# Export commonly used paths
REPO_ROOT = get_repo_root()
DATA_ROOT = get_data_root()
CACHE_ROOT = get_cache_root()

# Derived paths
EVID_DIR = CACHE_ROOT / "evidence"
PROGRESS_DIR = CACHE_ROOT / "progress"

# Dataset structure paths (optional, for backward compatibility)
# These are only used if configured in config.yaml
def get_real_dir() -> Optional[Path]:
    """Get real video directory if configured (for backward compatibility)"""
    try:
        config = load_config()
        structure = config.get("paths", {}).get("dataset_structure", {})
        real_dir = structure.get("real_dir")
        if real_dir:
            return DATA_ROOT / real_dir
    except Exception:
        pass
    return None

def get_fake_dir() -> Optional[Path]:
    """Get fake video directory if configured (for backward compatibility)"""
    try:
        config = load_config()
        structure = config.get("paths", {}).get("dataset_structure", {})
        fake_dir = structure.get("fake_dir")
        if fake_dir:
            return DATA_ROOT / fake_dir
    except Exception:
        pass
    return None

# For backward compatibility (deprecated, use get_real_dir()/get_fake_dir() instead)
REAL_DIR = get_real_dir()
FAKE_DIR = get_fake_dir()
