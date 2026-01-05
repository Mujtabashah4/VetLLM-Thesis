"""
VetLLM Training Script
Unified training script for Llama 3.1 8B and Qwen2.5 7B

Supports QLoRA fine-tuning with configurable parameters.
"""

import os
import sys
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML."""
    config_path: str
    _config: Dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict:
        return self._config.get('model', {})
    
    @property
    def quantization_config(self) -> Dict:
        return self._config.get('quantization', {})
    
    @property
    def lora_config(self) -> Dict:
        return self._config.get('lora', {})
    
    @property
    def training_config(self) -> Dict:
        return self._config.get('training', {})
    
    @property
    def data_config(self) -> Dict:
        return self._config.get('data', {})
    
    @property
    def wandb_config(self) -> Dict:
        return self._config.get('wandb', {})
    
    @property
    def experiment_config(self) -> Dict:
        return self._config.get('experiment', {})


def setup_wandb(config: TrainingConfig):
    """Initialize Weights & Biases if configured."""
    wandb_cfg = config.wandb_config
    if config.training_config.get('report_to') == 'wandb':
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get('project', 'vetllm'),
                name=wandb_cfg.get('name'),
                tags=wandb_cfg.get('tags', []),
                config=config._config,
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")


def get_quantization_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytesConfig for quantization."""
    quant_cfg = config.quantization_config
    
    if not quant_cfg.get('enabled', False):
        return None
    
    compute_dtype = getattr(torch, quant_cfg.get('bnb_4bit_compute_dtype', 'bfloat16'))
    
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_cfg.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get('bnb_4bit_use_double_quant', True),
    )


def get_lora_config(config: TrainingConfig) -> LoraConfig:
    """Create LoRA configuration."""
    lora_cfg = config.lora_config
    
    return LoraConfig(
        r=lora_cfg.get('r', 16),
        lora_alpha=lora_cfg.get('lora_alpha', 32),
        lora_dropout=lora_cfg.get('lora_dropout', 0.05),
        bias=lora_cfg.get('bias', 'none'),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """Load the model and tokenizer with quantization."""
    model_cfg = config.model_config
    model_name = model_cfg.get('name')
    
    # Check for local model first (faster loading)
    local_model_path = Path("models/llama3.1-8b-instruct")
    if local_model_path.exists() and (local_model_path / "config.json").exists():
        logger.info(f"Found local model at: {local_model_path}")
        logger.info(f"Using local model instead of downloading from HuggingFace")
        model_name = str(local_model_path.absolute())
    else:
        logger.info(f"Loading model from HuggingFace: {model_name}")
        logger.info("(Tip: Download model first with 'python download_llama3.1.py' for faster startup)")
    
    # Get quantization config
    bnb_config = get_quantization_config(config)
    
    # Determine torch dtype
    torch_dtype = getattr(torch, model_cfg.get('torch_dtype', 'bfloat16'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get('trust_remote_code', True),
        padding_side="right",
        token=True,  # Use HuggingFace token if needed
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    # Check if flash_attention_2 is available
    attn_impl = None
    if torch.cuda.is_available():
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            logger.warning("flash_attention_2 not available, using default attention")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch_dtype,  # Use dtype instead of deprecated torch_dtype
        device_map=config._config.get('hardware', {}).get('device_map', 'auto'),
        trust_remote_code=model_cfg.get('trust_remote_code', True),
        attn_implementation=attn_impl,
        token=True,  # Use HuggingFace token if needed
    )
    
    # Prepare model for k-bit training
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training_config.get('gradient_checkpointing', True),
        )
    
    # Apply LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_datasets(config: TrainingConfig, tokenizer):
    """Load and tokenize datasets."""
    data_cfg = config.data_config
    training_cfg = config.training_config
    
    # Get the base directory for data files
    config_dir = Path(config.config_path).parent.parent
    
    train_path = config_dir / data_cfg.get('train_file', 'data/train.json')
    val_path = config_dir / data_cfg.get('validation_file', 'data/validation.json')
    
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")
    
    # Load JSON files
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenization function
    text_field = data_cfg.get('text_field', 'text')
    max_length = training_cfg.get('max_seq_length', 512)
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
    
    # Tokenize datasets
    num_workers = data_cfg.get('preprocessing_num_workers', 4)
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=[col for col in train_dataset.column_names if col != 'input_ids'],
        desc="Tokenizing training data",
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=[col for col in val_dataset.column_names if col != 'input_ids'],
        desc="Tokenizing validation data",
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Create TrainingArguments from config."""
    training_cfg = config.training_config
    config_dir = Path(config.config_path).parent.parent
    
    output_dir = config_dir / training_cfg.get('output_dir', 'checkpoints')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle gradient checkpointing kwargs
    gc_kwargs = training_cfg.get('gradient_checkpointing_kwargs', {})
    
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.get('num_train_epochs', 3),
        per_device_train_batch_size=training_cfg.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_cfg.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 4),
        learning_rate=training_cfg.get('learning_rate', 1e-4),
        weight_decay=training_cfg.get('weight_decay', 0.01),
        warmup_ratio=training_cfg.get('warmup_ratio', 0.03),
        lr_scheduler_type=training_cfg.get('lr_scheduler_type', 'cosine'),
        gradient_checkpointing=training_cfg.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs=gc_kwargs if gc_kwargs else None,
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        eval_strategy=training_cfg.get('evaluation_strategy', training_cfg.get('eval_strategy', 'steps')),
        eval_steps=training_cfg.get('eval_steps', 100),
        save_strategy=training_cfg.get('save_strategy', 'steps'),
        save_steps=training_cfg.get('save_steps', 100),
        save_total_limit=training_cfg.get('save_total_limit', 3),
        load_best_model_at_end=training_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=training_cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_cfg.get('greater_is_better', False),
        logging_strategy=training_cfg.get('logging_strategy', 'steps'),
        logging_steps=training_cfg.get('logging_steps', 10),
        logging_first_step=training_cfg.get('logging_first_step', True),
        report_to=training_cfg.get('report_to', 'none'),
        optim=training_cfg.get('optim', 'paged_adamw_8bit'),
        fp16=training_cfg.get('fp16', False),
        bf16=training_cfg.get('bf16', True),
        seed=training_cfg.get('seed', 42),
        data_seed=training_cfg.get('data_seed', 42),
        remove_unused_columns=training_cfg.get('remove_unused_columns', False),
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 4),
        group_by_length=training_cfg.get('group_by_length', True),
    )


def train(config_path: str):
    """Main training function."""
    # Load configuration
    config = TrainingConfig(config_path)
    
    logger.info("=" * 60)
    logger.info(f"VetLLM Training - {config.experiment_config.get('name', 'experiment')}")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    seed = config.training_config.get('seed', 42)
    set_seed(seed)
    
    # Setup wandb
    setup_wandb(config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(config, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Get training arguments
    training_args = get_training_arguments(config)
    
    # Create trainer with early stopping if validation dataset exists
    callbacks = []
    if val_dataset is not None:
        early_stopping_patience = config.training_config.get('early_stopping_patience', 3)
        early_stopping_threshold = config.training_config.get('early_stopping_threshold', 0.001)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )
        logger.info(f"Early stopping enabled: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    
    # Train
    logger.info("Starting training...")
    # Check if we should resume from checkpoint
    resume_from_checkpoint = config.training_config.get('resume_from_checkpoint', None)
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    config_dir = Path(config_path).parent.parent
    final_output_dir = config_dir / "checkpoints" / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    # Save training metrics
    metrics = train_result.metrics
    metrics_path = final_output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete! Model saved to: {final_output_dir}")
    logger.info(f"Training metrics: {metrics}")
    
    return trainer, model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VetLLM Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    args = parser.parse_args()
    
    train(args.config)

