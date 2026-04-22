# Environment Setup Guide

## Local Machine (Windows)

### Prerequisites
- Python 3.11
- VS Code
- Git

### Installation
```bash
# Clone the repo
git clone https://github.com/coursdenatation/FNI-AI-LLM.git
cd FNI-AI-LLM

# Install dependencies
pip install -r requirements.txt
```

### Run the app
```bash
py -m src.year1.main
```

### Run tests
```bash
py -m pytest tests/ -v
```

### Run chat interfaces
```bash
# Terminal chat
py -m src.year1.chat.simple_chat
py -m src.year1.chat.nn_chat

# Browser UI
py -m streamlit run src/year1/chat/streamlit_app.py
```

---

## Google Colab (GPU Training)

PyTorch and Jupyter run in Colab — no local install needed.

### First time setup in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/FNI_AI_LLM
!git remote set-url origin https://coursdenatation:TOKEN@github.com/coursdenatation/FNI-AI-LLM.git
!git pull origin master
```

### Every session
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/FNI_AI_LLM
!git pull origin master
```

### Verify PyTorch in Colab
```python
import torch
print(torch.__version__)
print("GPU available:", torch.cuda.is_available())
!nvidia-smi
```

---

## Notes
- PyTorch is NOT installed locally (disk space constraint)
- All GPU training happens in Google Colab
- VS Code handles local development and testing
- GitHub syncs code between local and Colab
