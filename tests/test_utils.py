"""
Test suite for utility functions
"""

import pytest
from src.year1.utils import (
    get_data_path,
    get_model_path,
    create_directories,
    load_config,
    validate_data
)
import pandas as pd
from pathlib import Path

def test_get_data_path():
    """
    Test data path retrieval
    """
    path = get_data_path()
    assert isinstance(path, Path)
    assert str(path) == 'data'

def test_get_model_path():
    """
    Test model path retrieval
    """
    path = get_model_path()
    assert isinstance(path, Path)
    assert str(path) == 'models'

def test_create_directories(tmp_path):
    """
    Test directory creation
    """
    # Create test directories
    create_directories()
    
    # Check if directories exist
    assert (Path('data')).exists()
    assert (Path('models')).exists()
    assert (Path('tests/data')).exists()
    assert (Path('logs')).exists()

def test_load_config():
    """
    Test configuration loading
    """
    config = load_config()
    assert 'debug' in config
    assert 'log_level' in config
    assert 'data_path' in config
    assert 'model_path' in config

def test_validate_data():
    """
    Test data validation
    """
    # Test with valid data
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    assert validate_data(data) == True
    
    # Test with empty data
    with pytest.raises(ValueError):
        validate_data(pd.DataFrame())

if __name__ == "__main__":
    pytest.main()