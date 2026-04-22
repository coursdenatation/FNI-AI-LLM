@echo off
REM Initialize African Language Models project

echo Initializing African Language Models project...
echo.

REM Create virtual environment
python -m venv venv
echo Virtual environment created.

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install -e ".[dev]"
echo Dependencies installed.

REM Create directories
python -c "from src.year1.utils import create_directories; create_directories()"
echo Directories created.

REM Run initial tests
pytest tests/ -v
echo Tests completed.

REM Create logs directory
mkdir logs
echo Logs directory created.

echo.
echo Project initialization complete!
echo To start development, run: scripts\dev.bat
pause