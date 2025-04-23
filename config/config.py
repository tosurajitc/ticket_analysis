"""
Configuration management for the Incident Management Analytics application.
This module handles loading environment variables, config files, and providing
application settings to other modules.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from config.constants import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_API_TIMEOUT,
    DEFAULT_CACHE_TTL,
    SUPPORTED_FILE_TYPES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHART_TYPES,
    DEFAULT_THEME,
    DEFAULT_COLORS,
    MANDATORY_INCIDENT_COLUMNS,
    OPTIONAL_INCIDENT_COLUMNS,
    ANALYSIS_DIMENSIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the application configuration from environment variables and config files.
    
    Args:
        config_path: Optional path to a JSON config file
        
    Returns:
        Dict containing the application configuration
    """
    # Load environment variables
    load_dotenv()

    # Core configuration dictionary
    config = {
        "app_name": "Incident Management Analytics",
        "app_version": "1.0.0",
        "debug_mode": os.getenv("DEBUG_MODE", "False").lower() == "true",
        
        # LLM Configuration
        "llm": {
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model_name": os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            "max_tokens": int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            "temperature": float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
            "api_timeout": int(os.getenv("API_TIMEOUT", DEFAULT_API_TIMEOUT)),
            "cache_ttl": int(os.getenv("CACHE_TTL", DEFAULT_CACHE_TTL)),
        },
        
        # Data Processing Configuration
        "data": {
            "supported_file_types": SUPPORTED_FILE_TYPES,
            "chunk_size": int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            "mandatory_columns": MANDATORY_INCIDENT_COLUMNS,
            "optional_columns": OPTIONAL_INCIDENT_COLUMNS,
            "analysis_dimensions": ANALYSIS_DIMENSIONS,
            "max_sample_size": int(os.getenv("MAX_SAMPLE_SIZE", 10000)),
            "date_format": os.getenv("DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        },
        
        # Visualization Configuration
        "visualization": {
            "theme": os.getenv("THEME", DEFAULT_THEME),
            "colors": DEFAULT_COLORS,
            "chart_types": DEFAULT_CHART_TYPES,
            "max_categories": int(os.getenv("MAX_CATEGORIES", 10)),
            "default_height": int(os.getenv("DEFAULT_CHART_HEIGHT", 400)),
            "default_width": int(os.getenv("DEFAULT_CHART_WIDTH", 800)),
        },
        
        # Analysis Configuration
        "analysis": {
            "root_cause_depth": int(os.getenv("ROOT_CAUSE_DEPTH", 3)),
            "min_cluster_size": int(os.getenv("MIN_CLUSTER_SIZE", 5)),
            "anomaly_threshold": float(os.getenv("ANOMALY_THRESHOLD", 2.0)),
            "forecast_horizon": int(os.getenv("FORECAST_HORIZON", 14)),
            "max_automation_suggestions": int(os.getenv("MAX_AUTOMATION_SUGGESTIONS", 5)),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", 0.85)),
        },
        
        # UI Configuration
        "ui": {
            "page_size": int(os.getenv("PAGE_SIZE", 10)),
            "cache_expiry": int(os.getenv("CACHE_EXPIRY", 3600)),
            "enable_dark_mode": os.getenv("ENABLE_DARK_MODE", "True").lower() == "true",
            "enable_animations": os.getenv("ENABLE_ANIMATIONS", "True").lower() == "true",
            "enable_export": os.getenv("ENABLE_EXPORT", "True").lower() == "true",
            "enable_filtering": os.getenv("ENABLE_FILTERING", "True").lower() == "true",
        },
    }
    
    # Load configuration from JSON file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Deep merge with default config
                config = deep_update(config, file_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Validate configuration
    _validate_config(config)
    
    return config

def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration and set reasonable defaults if needed.
    
    Args:
        config: Configuration dictionary to validate
    """
    # Validate token limits
    if config["llm"]["max_tokens"] > 5000:
        logger.warning("max_tokens exceeds 5000, setting to 5000")
        config["llm"]["max_tokens"] = 5000
    
    # Validate temperature
    if not 0 <= config["llm"]["temperature"] <= 1:
        logger.warning(f"Invalid temperature value: {config['llm']['temperature']}, setting to default")
        config["llm"]["temperature"] = DEFAULT_TEMPERATURE
    
    # Validate mandatory columns
    if not config["data"]["mandatory_columns"]:
        logger.error("No mandatory columns defined, using defaults")
        config["data"]["mandatory_columns"] = MANDATORY_INCIDENT_COLUMNS
    
    # Validate API key existence
    if not config["llm"]["api_key"]:
        logger.warning("No API key provided. User will need to input key through UI.")

def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary with values from another dictionary.
    
    Args:
        original: Original dictionary to update
        update: Dictionary with values to update the original
        
    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

def get_api_key(config: Dict[str, Any], user_provided_key: Optional[str] = None) -> str:
    """
    Get the API key to use for LLM requests.
    
    Args:
        config: Application configuration
        user_provided_key: API key provided by the user through the UI
        
    Returns:
        API key to use
    """
    # Priority: user provided key > config
    if user_provided_key:
        return user_provided_key
    
    return config["llm"]["api_key"]

# Config class for the application
class Config:
    """
    Comprehensive configuration management for the Incident Management Analytics application.
    """
    _instance = None

    def __new__(cls):
        """
        Singleton pattern implementation
        """
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize configuration settings
        """
        # Load environment variables
        load_dotenv()

        # Core configuration dictionary
        self._config = load_config()
        
        # Add attribute access for main configuration keys
        self.app_name = self._config["app_name"]
        self.app_version = self._config["app_version"]
        self.debug_mode = self._config["debug_mode"]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated configuration key
            default: Default value if key is not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_api_key(self, user_provided_key: Optional[str] = None) -> str:
        """
        Get the API key to use for LLM requests.
        
        Args:
            user_provided_key: API key provided by the user through the UI
        
        Returns:
            API key to use
        """
        return get_api_key(self._config, user_provided_key)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the full configuration as a dictionary.
        
        Returns:
            Full configuration dictionary
        """
        return self._config.copy()
    
    def __getitem__(self, key):
        """
        Allow dictionary-style access (config['key']) for backward compatibility
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self._config[key]

# Create a singleton instance
config = Config()