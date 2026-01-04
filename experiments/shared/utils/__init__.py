"""
VetLLM Shared Utilities
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = Path(__file__).parent.parent.parent
SHARED_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset_UVAS"

# Model identifiers
MODELS = {
    "llama3.1": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "display_name": "Llama 3.1 8B Instruct",
        "chat_format": "llama3.1",
    },
    "qwen2.5": {
        "name": "Qwen/Qwen2.5-7B-Instruct", 
        "display_name": "Qwen2.5 7B Instruct",
        "chat_format": "qwen2.5",
    },
    "qwen2.5-medical": {
        "name": "HPAI-BSC/Qwen2.5-Aloe-Beta-7B",
        "display_name": "Qwen2.5 Aloe Beta 7B (Medical)",
        "chat_format": "qwen2.5",
    },
}

__all__ = [
    "PROJECT_ROOT",
    "EXPERIMENTS_DIR", 
    "SHARED_DIR",
    "DATASET_DIR",
    "MODELS",
]

