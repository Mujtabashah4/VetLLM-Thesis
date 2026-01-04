"""
VetLLM Evaluation Module
"""

from .evaluate import (
    EvaluationConfig,
    MetricsCalculator,
    evaluate,
    extract_diagnosis_from_output,
    normalize_diagnosis_name,
)

__all__ = [
    "EvaluationConfig",
    "MetricsCalculator",
    "evaluate",
    "extract_diagnosis_from_output",
    "normalize_diagnosis_name",
]

