@echo off
REM Setup script for African Language Models project

echo Setting up African Language Models project...
echo.

REM Create virtual environment
python -m venv venv
echo Virtual environment created.

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install -e ".[dev]"
echo Dependencies installed.

REM Run initial tests
pytest tests/ -v
echo Tests completed.

echo.
echo Setup complete!
echo To start development, run: scripts\dev.bat
pause