"""
Utility functions for African Language Models project
"""

import os
import logging
from pathlib import Path
from .colab_utils import is_colab, setup_colab_env, get_drive_prefix

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_data_path():
    """
    Get the path to the data directory
    """
    if is_colab():
        # Automatically point to Drive when in Colab
        return get_drive_prefix() / 'FNI_AI_LLM' / 'data'
    
    return Path(os.getenv('DATA_PATH', 'data'))

def get_model_path():
    """
    Get the path to the models directory
    """
    if is_colab():
        # Automatically point to Drive when in Colab
        return get_drive_prefix() / 'FNI_AI_LLM' / 'models'

    return Path(os.getenv('MODEL_PATH', 'models'))

def create_directories():
    """
    Create necessary directories if they don't exist
    """
    directories = [
        get_data_path(),
        get_data_path() / 'raw',
        get_data_path() / 'processed',
        get_data_path() / 'african_languages',
        get_model_path(),
        get_model_path() / 'checkpoints',
        Path('tests/data'),
        Path('logs'),
        Path('notebooks/experiments'),
        Path('docs/visualizations')
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def load_config():
    """
    Load configuration from environment variables
    """
    # Auto-setup if in Colab environment
    if is_colab():
        setup_colab_env()

    config = {
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'data_path': get_data_path(),
        'model_path': get_model_path()
    }
    return config

def validate_data(data):
    """
    Validate data integrity
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or invalid")
    return True