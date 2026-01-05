#!/usr/bin/env python3
"""
Improved Training Script for VetLLM with:
- More epochs (5-10) with early stopping
- Train/Validation split
- Validation monitoring
- Best model checkpointing
- Overfitting prevention
"""
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImprovedVetLLMConfig:
    """Improved configuration with validation and early stopping"""

    # Model configuration
    model_name: str = "models/alpaca-7b-native"
    max_length: int = 512
    use_lora: bool = True

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training configuration - IMPROVED (CONTINUED TRAINING)
    num_epochs: int = 7  # Reasonable: 7 epochs = ~1,120 total steps (500 + 620 more)
    batch_size: int = 128
    per_device_batch_size: int = 2  # RTX 4090 optimized
    gradient_accumulation_steps: int = 4  # Effective batch = 2 * 4 = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Early Stopping - RELAXED for continued training
    early_stopping_patience: int = 5  # Increased patience to allow more training
    early_stopping_threshold: float = 0.0005  # Lower threshold for continued improvement

    # Optimization
    bf16: bool = True  # RTX 4090 optimized
    fp16: bool = False
    use_4bit: bool = True  # QLoRA
    tf32: bool = True
    gradient_checkpointing: bool = False

    # Evaluation and saving - IMPROVED
    eval_strategy: str = "steps"  # Changed from "no" to "steps"
    eval_steps: int = 50  # Evaluate every 50 steps
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3  # Keep 3 best checkpoints
    load_best_model_at_end: bool = True  # Load best model at end
    metric_for_best_model: str = "eval_loss"  # Use validation loss
    greater_is_better: bool = False  # Lower loss is better

    # Data split - NEW
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Paths
    data_path: str = "processed_data/all_processed_data_augmented.json"  # Use augmented data with rare diseases
    output_dir: str = "models/vetllm-finetuned-continued"
    cache_dir: str = "cache"
    resume_from_checkpoint: Optional[str] = "models/vetllm-finetuned-correct/checkpoint-500"  # Resume from here

    # Logging
    use_wandb: bool = False
    logging_steps: int = 10


class ImprovedVetLLMDataProcessor:
    """Data processing with train/val split"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def create_alpaca_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Create Alpaca-style prompt (MATCHING ORIGINAL FORMAT)"""
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

    def split_data(self, data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Split data into train/val/test sets"""
        logger.info(f"Loading data from {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Total samples: {len(data)}")

        # Shuffle data
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        data = [data[i] for i in indices]

        # Calculate split sizes
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        logger.info(f"Train: {len(train_data)} samples")
        logger.info(f"Validation: {len(val_data)} samples")
        logger.info(f"Test: {len(test_data)} samples")

        return train_data, val_data, test_data

    def prepare_dataset(self, data: List[dict]) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Preparing dataset with {len(data)} samples")

        formatted_data = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            prompt = self.create_alpaca_prompt(instruction, input_text, output)
            formatted_data.append({"text": prompt})

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize without returning tensors - let the data collator handle batching
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Let data collator handle padding
                return_tensors=None,  # Return lists, not tensors
            )
            # Don't create labels here - let DataCollatorForLanguageModeling handle it
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,  # Remove all original columns, keep only tokenized ones
            desc="Tokenizing dataset",
        )

        return tokenized_dataset


class ImprovedVetLLMTrainer:
    """Improved trainer with validation and early stopping"""

    def __init__(self, config: ImprovedVetLLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization"""
        logger.info(f"Loading model from {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 4-bit quantization
        if self.config.use_4bit and torch.cuda.is_available():
            logger.info("Loading model with 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        else:
            model_dtype = torch.bfloat16 if self.config.bf16 else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=model_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        logger.info("Model loaded successfully")

    def setup_lora(self):
        """Setup LoRA adapters"""
        if not self.config.use_lora:
            return

        logger.info("Setting up LoRA adapters")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def create_training_arguments(self, train_dataset_size: int) -> TrainingArguments:
        """Create training arguments with early stopping"""
        # Calculate total steps
        steps_per_epoch = train_dataset_size // (
            self.config.per_device_batch_size * self.config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            tf32=self.config.tf32,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=self.config.output_dir.split("/")[-1],
        )

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train model with validation"""
        logger.info("=" * 80)
        logger.info("STARTING IMPROVED TRAINING WITH VALIDATION")
        logger.info("=" * 80)

        # Create training arguments
        training_args = self.create_training_arguments(len(train_dataset))

        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Create trainer with early stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            ],
        )

        # Train
        logger.info("Starting training...")
        resume_from = self.config.resume_from_checkpoint if hasattr(self.config, 'resume_from_checkpoint') and self.config.resume_from_checkpoint else None
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming training from checkpoint: {resume_from}")
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()

        # Log metrics
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        logger.info(f"Best validation loss: {trainer.state.best_metric:.4f}")
        logger.info(f"Total steps: {train_result.global_step}")
        logger.info(f"Model saved to: {self.config.output_dir}")

        return train_result


def main():
    """Main training function"""
    config = ImprovedVetLLMConfig()

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize components
    trainer = ImprovedVetLLMTrainer(config)
    trainer.load_model_and_tokenizer()
    trainer.setup_lora()

    # Process data
    processor = ImprovedVetLLMDataProcessor(trainer.tokenizer, config.max_length)

    # Split data
    train_data, val_data, test_data = processor.split_data(
        config.data_path,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
    )

    # Prepare datasets
    train_dataset = processor.prepare_dataset(train_data)
    val_dataset = processor.prepare_dataset(val_data)

    # Train
    train_result = trainer.train(train_dataset, val_dataset)

    logger.info("✅ Training complete!")
    logger.info(f"Best model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

