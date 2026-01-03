#!/usr/bin/env python3
"""
VetLLM Training Script
Complete implementation for fine-tuning Alpaca-7B for veterinary diagnosis prediction
"""

import os
import json
import torch
import yaml
import wandb
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import argparse
import logging

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check DeepSpeed availability
if not DEEPSPEED_AVAILABLE:
    logger.warning(
        "DeepSpeed not available. Training will continue without DeepSpeed optimization."
    )

@dataclass
class VetLLMConfig:
    """Configuration for VetLLM training"""

    # Model configuration
    model_name: str = "wxjiao/alpaca-7b"
    max_length: int = 512
    use_lora: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training configuration
    num_epochs: int = 3
    batch_size: int = 128
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 4 * 4 = 16 (matches notebook)
    learning_rate: float = 2e-4  # Higher LR for LoRA (matches notebook)
    weight_decay: float = 0.01  # Matches notebook
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Optimization
    bf16: bool = False  # Disabled by default for MPS compatibility
    fp16: bool = True  # Use FP16 mixed precision for CUDA (faster training)
    use_8bit: bool = False  # Use 8-bit quantization (disabled by default for full precision)
    tf32: bool = True  # Enable TF32 for Ampere+ GPUs (faster)
    gradient_checkpointing: bool = True
    deepspeed_config: Optional[str] = None

    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 50  # More frequent evaluation (matches notebook)
    save_strategy: str = "steps"
    save_steps: int = 100  # More frequent saving (matches notebook)
    save_total_limit: int = 2  # Keep 2 checkpoints (matches notebook)

    # Paths
    data_path: str = "data/processed/train_data.json"
    val_data_path: str = "data/processed/val_data.json"
    output_dir: str = "models/vetllm-finetuned"
    cache_dir: str = "cache"

    # Logging
    use_wandb: bool = True
    project_name: str = "vetllm-training"
    run_name: str = None

class VetLLMDataProcessor:
    """Data processing for veterinary instruction tuning"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def create_alpaca_prompt(
        self, instruction: str, input_text: str = "", output: str = ""
    ) -> str:
        """Create Alpaca-style prompt"""
        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        return prompt

    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Loading dataset from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} samples")

        # Format all data
        formatted_data = []
        for item in data:
            prompt = self.create_alpaca_prompt(
                item["instruction"], item.get("input", ""), item["output"]
            )
            formatted_data.append({"text": prompt})

        dataset = Dataset.from_list(formatted_data)

        def tokenize_function(examples):
            # Tokenize without returning tensors - let the data collator handle batching
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Let data collator handle padding
                return_tensors=None,  # Return lists, not tensors
            )
            # Don't create labels here - let DataCollatorForLanguageModeling handle it
            return tokenized

        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove all original columns, keep only tokenized ones
        )
        logger.info("Dataset tokenization completed")

        return tokenized_dataset

class VetLLMTrainer:
    """Main trainer class for VetLLM"""

    def __init__(self, config: VetLLMConfig):
        self.config = config
        self.setup_logging()
        self.load_model_and_tokenizer()
        self.data_processor = VetLLMDataProcessor(self.tokenizer, config.max_length)

    def setup_logging(self):
        """Setup logging and monitoring"""
        if not os.path.exists("logs"):
            os.makedirs("logs")

        # Initialize wandb if available and enabled
        if self.config.use_wandb:
            try:
                run_name = (
                    self.config.run_name
                    or f"vetllm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                wandb.init(
                    project=self.config.project_name,
                    config=self.config.__dict__,
                    name=run_name,
                )
                self.use_wandb = True
                logger.info("Wandb initialized successfully")
            except Exception as e:
                logger.warning(f"Wandb initialization failed: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def load_model_and_tokenizer(self):
        """Load and configure model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer with explicit handling for LLaMA models
        try:
            # Try to import sentencepiece first
            import sentencepiece as spm

            logger.info("SentencePiece library available")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
                use_fast=False,  # Force slow tokenizer to avoid conversion issues
            )
        except ImportError as e:
            logger.error(f"SentencePiece not available: {e}")
            raise ImportError(
                "SentencePiece is required for LLaMA tokenizers. Please install it with: pip install sentencepiece"
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer with default settings: {e}")
            try:
                # Try with minimal settings
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name, use_fast=False, legacy=False
                )
            except Exception as e2:
                logger.error(f"Failed to load tokenizer with fallback settings: {e2}")
                raise e2

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Tokenizer loaded successfully")

        # Determine the best device for the current system
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        self.device = device

        # Load model with 8-bit quantization if enabled and CUDA available
        if self.config.use_8bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                
                logger.info("Loading model with 8-bit quantization for memory efficiency...")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                logger.info("Model loaded with 8-bit quantization")
                
                # Prepare model for k-bit training (CRITICAL for 8-bit)
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Model prepared for k-bit training")
                
            except ImportError:
                logger.warning("bitsandbytes not available. Falling back to standard loading.")
                self.config.use_8bit = False
            except Exception as e:
                logger.warning(f"8-bit quantization failed: {e}. Falling back to standard loading.")
                self.config.use_8bit = False
        
        # Fallback to standard loading
        if not self.config.use_8bit or not torch.cuda.is_available():
            import os
            offload_folder = os.path.join(self.config.output_dir, "offload")
            os.makedirs(offload_folder, exist_ok=True)
            
            # Determine dtype
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                model_dtype = torch.float32
                device_map = None
            else:
                model_dtype = torch.float16 if self.config.bf16 else torch.float32
                device_map = "auto" if torch.cuda.is_available() else None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=model_dtype,
                device_map=device_map,
                trust_remote_code=True,
                offload_folder=offload_folder if device_map else None,
                low_cpu_mem_usage=True,
            )

            # Move model to device if not using device_map
            if device_map is None:
                self.model = self.model.to(device)

        logger.info("Base model loaded successfully")

        # Setup LoRA if enabled (do this after model loading)
        if self.config.use_lora:
            self.setup_lora()

        # Note: Gradient checkpointing is handled in TrainingArguments
        # Don't enable it here for 8-bit models as it may conflict

    def setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        # Get PEFT model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters info
        self.model.print_trainable_parameters()

        # Enable training mode
        self.model.train()
        
        # Explicitly enable gradients for LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                # Ensure parameter is on the correct device and dtype
                if hasattr(self, 'device'):
                    param.data = param.data.to(self.device)
        
        # Verify trainable parameters
        trainable_params = []
        total_trainable = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
                total_trainable += param.numel()
        
        logger.info(f"Number of trainable parameter tensors: {len(trainable_params)}")
        logger.info(f"Total trainable parameters: {total_trainable:,}")
        
        if total_trainable == 0:
            raise ValueError("No trainable parameters found after LoRA setup!")
        
        # Log first few trainable parameters for debugging
        for name, count in trainable_params[:5]:
            logger.info(f"  {name}: {count:,} parameters")

        if not trainable_params:
            logger.error(
                "No trainable parameters found! This will cause the gradient error."
            )
            raise ValueError("No trainable parameters found in the model")

        logger.info("LoRA setup completed")

    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""

        # Adjust settings based on device
        use_bf16 = self.config.bf16
        use_mps_device = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        
        if use_mps_device:
            # Disable bf16 on MPS for stability
            use_bf16 = False
            logger.info("Disabling bf16 for MPS compatibility")

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            # Optimization settings
            bf16=use_bf16,
            fp16=self.config.fp16 and not self.config.use_8bit and torch.cuda.is_available(),  # FP16 for CUDA (matches notebook)
            tf32=self.config.tf32 and not use_mps_device,  # Disable TF32 on MPS
            gradient_checkpointing=self.config.gradient_checkpointing and not self.config.use_8bit,  # Disable for 8-bit
            # Evaluation and saving
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            # Logging
            logging_steps=10,
            logging_dir=f"{self.config.output_dir}/logs",
            report_to="wandb" if self.use_wandb else "none",
            # Other settings
            dataloader_drop_last=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            # MPS specific optimizations
            use_cpu=(not torch.cuda.is_available() and not use_mps_device),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,  # Disable pin memory for MPS
        )

        # Add DeepSpeed config if specified and available
        if self.config.deepspeed_config and DEEPSPEED_AVAILABLE:
            args.deepspeed = self.config.deepspeed_config
        elif self.config.deepspeed_config and not DEEPSPEED_AVAILABLE:
            logger.warning(
                "DeepSpeed config specified but DeepSpeed is not available. Ignoring DeepSpeed configuration."
            )

        return args

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Train the model"""
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Create training arguments
        training_args = self.create_training_arguments()

        # Data collator
        # Pad to multiple of 8 for FP16/BF16 efficiency
        use_mixed_precision = (self.config.fp16 or self.config.bf16) and torch.cuda.is_available()
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if use_mixed_precision else None,
        )

        # Ensure model is in training mode
        self.model.train()

        # Final verification of trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Total trainable parameters: {trainable_params:,}")

        if trainable_params == 0:
            logger.error("No trainable parameters found!")
            raise ValueError("Model has no trainable parameters")
        
        # Additional check for gradient computation
        requires_grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                requires_grad_params.append(name)
        
        logger.info(f"Parameters requiring gradients: {len(requires_grad_params)}")
        if len(requires_grad_params) == 0:
            raise ValueError("No parameters are set to require gradients!")
        
        # Log first few parameters that require gradients
        for name in requires_grad_params[:5]:
            logger.info(f"  {name} requires grad: {self.model.get_parameter(name).requires_grad}")
        
        # Ensure model is prepared for training on the correct device
        if hasattr(self, 'device'):
            self.model = self.model.to(self.device)
        
        # Force enable training mode for all trainable modules
        for name, module in self.model.named_modules():
            if any(p.requires_grad for p in module.parameters()):
                module.train()

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=(
                [EarlyStoppingCallback(early_stopping_patience=3)]
                if eval_dataset
                else None
            ),
        )

        # Train the model
        logger.info("=" * 50)
        logger.info("TRAINING STARTED")
        logger.info("=" * 50)

        train_result = trainer.train()

        # Save the model
        trainer.save_model()
        trainer.save_state()

        # Log training results
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        logger.info(f"Training steps: {train_result.global_step}")
        logger.info(f"Model saved to: {self.config.output_dir}")

        if self.use_wandb:
            wandb.log(
                {
                    "final_train_loss": train_result.training_loss,
                    "training_steps": train_result.global_step,
                }
            )
            wandb.finish()

        return trainer

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="VetLLM Training")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument(
        "--model-name", default="wxjiao/alpaca-7b", help="Base model name"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/train_data.json",
        help="Training data path",
    )
    parser.add_argument(
        "--val-data-path",
        default="data/processed/val_data.json",
        help="Validation data path",
    )
    parser.add_argument(
        "--output-dir", default="models/vetllm-finetuned", help="Output directory"
    )
    parser.add_argument(
        "--use-lora", action="store_true", default=True, help="Use LoRA fine-tuning"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Create config
    config = VetLLMConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
    )

    logger.info("VetLLM Training Pipeline")
    logger.info("=" * 50)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Use LoRA: {config.use_lora}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.per_device_batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 50)

    # Create trainer
    trainer = VetLLMTrainer(config)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = trainer.data_processor.prepare_dataset(config.data_path)

    eval_dataset = None
    if os.path.exists(config.val_data_path):
        eval_dataset = trainer.data_processor.prepare_dataset(config.val_data_path)

    # Train the model
    trained_model = trainer.train(train_dataset, eval_dataset)

    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
