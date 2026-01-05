"""
Custom Trainer with Class-Weighted Loss
Implements weighted loss function to handle class imbalance
"""

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced datasets."""
    
    def __init__(self, class_weights: Optional[Dict[str, float]] = None, *args, **kwargs):
        """
        Initialize weighted trainer.
        
        Args:
            class_weights: Dictionary mapping disease names to weights
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights or {}
        self.weight_tensor = None
        
        if class_weights:
            logger.info(f"Class weights initialized: {len(class_weights)} diseases")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute weighted loss.
        
        For language modeling, we compute loss on the entire sequence.
        The weighting is applied based on the disease label in metadata.
        """
        # Get standard loss
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute per-token loss
        per_token_loss = loss_fct(shift_logits, shift_labels)
        
        # Reshape to (batch_size, sequence_length)
        per_token_loss = per_token_loss.view(labels.size(0), -1)
        
        # Average over sequence length (ignoring padding)
        valid_tokens = (labels != -100).float()
        per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
        
        # Apply class weights if available
        if self.class_weights and hasattr(self, 'current_diseases'):
            # Get disease labels for current batch
            weights = []
            for disease in self.current_diseases:
                weight = self.class_weights.get(disease, 1.0)
                weights.append(weight)
            
            if weights:
                weight_tensor = torch.tensor(weights, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
                per_sample_loss = per_sample_loss * weight_tensor
        
        # Average over batch
        loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss


def calculate_class_weights(data_path: str) -> Dict[str, float]:
    """
    Calculate class weights inversely proportional to frequency.
    
    Args:
        data_path: Path to training JSON file
        
    Returns:
        Dictionary mapping disease names to weights
    """
    import json
    from collections import Counter
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Count diseases
    disease_counts = Counter()
    for item in data:
        metadata = item.get('metadata', {})
        disease = metadata.get('disease_normalized') or metadata.get('disease', 'Unknown')
        if disease and disease != 'Unknown':
            disease_counts[disease] += 1
    
    # Calculate weights (inverse frequency)
    total_samples = sum(disease_counts.values())
    max_count = max(disease_counts.values())
    
    weights = {}
    for disease, count in disease_counts.items():
        # Weight = max_count / count (inverse frequency)
        # Normalize so most common disease has weight 1.0
        weight = max_count / count if count > 0 else max_count
        weights[disease] = float(weight)
    
    logger.info(f"Calculated class weights for {len(weights)} diseases")
    logger.info(f"Weight range: {min(weights.values()):.2f} - {max(weights.values()):.2f}")
    
    return weights

