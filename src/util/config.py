"""Configuration loader from YAML and environment variables"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()


def get_repo_root() -> Path:
    """Get repository root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, override with env vars.

    Args:
        config_path: Config file name (e.g., "config.yaml", "ds.yaml").
                    If None, uses "config.yaml".
                    Only filename is supported, not full paths.
                    
    Returns:
        Configuration dict. Missing required fields will cause errors when accessed.
    """
    # Get config directory
    current_file = Path(__file__).resolve()
    config_dir = current_file.parent.parent / "config"

    # Resolve config path
    if config_path is None:
        # Default to config.yaml
        config_filename = "config.yaml"
    else:
        # Only accept filename, not paths
        config_filename = Path(config_path).name
        if "/" in config_path or "\\" in config_path:
            raise ValueError(
                f"配置路径应仅为文件名，不支持路径: {config_path}。"
                f"请使用文件名，如 'ds.yaml' 或 'config.yaml'"
            )

    config_path_obj = config_dir / config_filename

    # Load YAML config (will raise FileNotFoundError if not exists)
    with open(config_path_obj, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Override with env vars
    if os.getenv("DATA_ROOT"):
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["data_root"] = os.getenv("DATA_ROOT")

    if os.getenv("CACHE_ROOT"):
        if "paths" not in config:
            config["paths"] = {}
        config["paths"]["cache_root"] = os.getenv("CACHE_ROOT")

    if os.getenv("LLM_PROVIDER"):
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["provider"] = os.getenv("LLM_PROVIDER")

    if os.getenv("OPENAI_MODEL"):
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["model"] = os.getenv("OPENAI_MODEL")



    return config


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by dot-separated path."""
    config = load_config()
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# Convenience functions for common config values
def get_frame_ext() -> str:
    return get_config_value("video.frame_ext", "jpg")


def get_frame_quality() -> int:
    return get_config_value("video.frame_quality", 2)


def get_frame_mime() -> str:
    return get_config_value("video.frame_mime", "image/jpeg")


def get_lock_wait_sec() -> float:
    return get_config_value("concurrency.lock_wait_sec", 8.0)


def get_lock_poll_sec() -> float:
    return get_config_value("concurrency.lock_poll_sec", 0.25)


def get_batch_delay_sec() -> float:
    return get_config_value("batch.delay_sec", 0.1)
