"""
Configuration loader for the Drunk Detection System.

Loads YAML config files and supports environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override with environment variables where applicable
    config = _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config values with environment variables.

    Environment variable mapping:
        TELEGRAM_TOKEN      -> deployment.telegram_token
        TELEGRAM_CHAT_ID    -> deployment.telegram_chat_id
        SERIAL_PORT         -> deployment.serial_port
        SERIAL_BAUDRATE     -> deployment.serial_baudrate
        MLFLOW_TRACKING_URI -> mlflow.tracking_uri
    """
    env_mapping = {
        "TELEGRAM_TOKEN": ("deployment", "telegram_token"),
        "TELEGRAM_CHAT_ID": ("deployment", "telegram_chat_id"),
        "SERIAL_PORT": ("deployment", "serial_port"),
        "SERIAL_BAUDRATE": ("deployment", "serial_baudrate"),
        "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    }

    for env_var, (section, key) in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            if section not in config:
                config[section] = {}
            config[section][key] = value

    return config


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested config value.

    Args:
        config: Configuration dictionary.
        *keys: Nested key path (e.g., "training", "learning_rate").
        default: Default value if key not found.

    Returns:
        Config value or default.

    Example:
        >>> cfg = load_config()
        >>> lr = get_config_value(cfg, "training", "learning_rate", default=1e-4)
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
