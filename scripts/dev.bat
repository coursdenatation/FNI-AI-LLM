@echo off
REM Development script for African Language Models project (Windows)

REM Activate virtual environment
call venv\Scripts\activate

REM Install development dependencies
pip install -e ".[dev]"

REM Run development server
python -m src.year1.main

REM Run tests
pytest tests/

REM Run code formatting
black src/ tests/

REM Run linting
flake8 src/ tests/