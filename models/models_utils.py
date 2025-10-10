#!/usr/bin/env python3
"""
Model Utilities for VetLLM Pipeline
Helper functions for model loading, saving, and management
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Utility class for managing VetLLM models"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def load_base_model(self, model_name: str = "wxjiao/alpaca-7b") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load base Alpaca model and tokenizer"""
        logger.info(f"Loading base model: {model_name}")
        
        # Check if model exists locally
        local_path = self.models_dir / "alpaca-7b"
        if local_path.exists() and any(local_path.iterdir()):
            model_path = str(local_path)
            logger.info(f"Loading from local path: {model_path}")
        else:
            model_path = model_name
            logger.info(f"Loading from Hugging Face: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("Base model loaded successfully")
        return model, tokenizer
    
    def load_finetuned_model(self, model_path: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load fine-tuned VetLLM model"""
        if model_path is None:
            model_path = self.models_dir / "vetllm-finetuned"
            
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
        
        logger.info(f"Loading fine-tuned model from: {model_path}")
        
        # Check if it's a PEFT model
        adapter_config_path = model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            # Load as PEFT model
            logger.info("Loading as PEFT/LoRA model")
            
            # Load adapter config to get base model
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "wxjiao/alpaca-7b")
            
            # Load base model first
            base_model, tokenizer = self.load_base_model(base_model_name)
            
            # Load PEFT model
            model = PeftModel.from_pretrained(base_model, str(model_path))
            
        else:
            # Load as full model
            logger.info("Loading as full fine-tuned model")
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        model.eval()
        logger.info("Fine-tuned model loaded successfully")
        return model, tokenizer
    
    def save_model_info(self, model_path: str, info: Dict):
        """Save model information to JSON file"""
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        info_file = model_path / "model_config.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model info saved to {info_file}")
    
    def get_model_info(self, model_path: str) -> Dict:
        """Load model information from JSON file"""
        model_path = Path(model_path)
        info_file = model_path / "model_config.json"
        
        if not info_file.exists():
            return {}
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def list_available_models(self) -> List[Dict]:
        """List all available models in the models directory"""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                info = self.get_model_info(str(model_dir))
                if info:
                    info["path"] = str(model_dir)
                    models.append(info)
                else:
                    # Basic info if no config file
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": "unknown"
                    })
        
        return models
    
    def check_model_compatibility(self, model_path: str) -> Dict:
        """Check model compatibility and requirements"""
        model_path = Path(model_path)
        
        compatibility = {
            "exists": model_path.exists(),
            "has_config": (model_path / "config.json").exists(),
            "has_tokenizer": (model_path / "tokenizer.json").exists() or (model_path / "tokenizer.model").exists(),
            "is_peft": (model_path / "adapter_config.json").exists(),
            "estimated_size_gb": 0,
            "requirements": []
        }
        
        if compatibility["exists"]:
            # Calculate size
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            compatibility["estimated_size_gb"] = total_size / (1024**3)
            
            # Check requirements
            if compatibility["estimated_size_gb"] > 10:
                compatibility["requirements"].append("GPU with >16GB VRAM recommended")
            
            if compatibility["is_peft"]:
                compatibility["requirements"].append("PEFT library required")
                compatibility["memory_efficient"] = True
            else:
                compatibility["memory_efficient"] = False
        
        return compatibility

def main():
    """Demo function for model utilities"""
    manager = ModelManager()
    
    print("VetLLM Model Manager Demo")
    print("=" * 40)
    
    # List available models
    models = manager.list_available_models()
    print(f"Available models: {len(models)}")
    for model in models:
        print(f"  - {model.get('name', 'Unknown')}: {model['path']}")
    
    # Check base model
    base_path = "./models/alpaca-7b"
    if Path(base_path).exists():
        compatibility = manager.check_model_compatibility(base_path)
        print(f"\nBase model compatibility:")
        print(f"  Size: {compatibility['estimated_size_gb']:.1f} GB")
        print(f"  Requirements: {compatibility['requirements']}")
    
    # Check fine-tuned model
    finetuned_path = "./models/vetllm-finetuned"
    if Path(finetuned_path).exists():
        compatibility = manager.check_model_compatibility(finetuned_path)
        print(f"\nFine-tuned model compatibility:")
        print(f"  Size: {compatibility['estimated_size_gb']:.1f} GB")
        print(f"  PEFT model: {compatibility['is_peft']}")
        print(f"  Memory efficient: {compatibility['memory_efficient']}")

if __name__ == "__main__":
    main()
