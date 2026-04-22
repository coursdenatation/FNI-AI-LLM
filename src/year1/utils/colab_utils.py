"""
Utilities for Google Colab integration
"""
import os
import sys
from pathlib import Path


def is_colab():
    """
    Check if the code is running in a Google Colab environment
    """
    return 'google.colab' in sys.modules


def get_drive_prefix():
    """
    Returns the standard Google Drive mount prefix
    """
    return Path('/content/drive/MyDrive')


def setup_colab_env(project_folder_name='FNI_AI_LLM', mount_drive=True):
    """
    Sets up the environment for Colab, including mounting drive and pathing
    """
    if not is_colab():
        return False

    if mount_drive:
        from google.colab import drive
        print("Detected Google Colab environment. Mounting Drive...")
        drive.mount('/content/drive')

    project_root = get_drive_prefix() / project_folder_name
    if os.path.exists(project_root):
        os.chdir(project_root)
        sys.path.append(str(project_root))
        print(f"Changed directory to {project_root} and added to sys.path")
    else:
        print(f"Warning: Project root {project_root} not found.")
    return True
