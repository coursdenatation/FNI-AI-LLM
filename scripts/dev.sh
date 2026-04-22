#!/bin/bash

# Development script for African Language Models project

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run development server
python -m src.year1.main

# Run tests
pytest tests/

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/