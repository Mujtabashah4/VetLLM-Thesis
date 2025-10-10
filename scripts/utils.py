"""
VetLLM Utility Functions
Common helpers for preprocessing, validation, and metrics.
"""

import os
from pathlib import Path

def ensure_dir(path):
    """Create directory if not exists"""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def read_json_file(path):
    """Load JSON from path"""
    import json
    with open(path, "r") as f:
        return json.load(f)

def write_json_file(obj, path):
    """Write JSON to path"""
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
