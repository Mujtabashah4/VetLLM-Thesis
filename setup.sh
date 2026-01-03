#!/bin/bash
# VetLLM Setup Script
# Installs all dependencies and prepares the environment

set -e  # Exit on error

echo "=========================================="
echo "VetLLM Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $python_version"

# Check if CUDA is available (optional)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo ""
    echo "⚠️  No NVIDIA GPU detected. Training will use CPU (very slow)."
fi

echo ""
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'✅ PEFT: {peft.__version__}')"
python3 -c "import datasets; print(f'✅ Datasets: {datasets.__version__}')"

echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "Validating data files..."
python3 scripts/validate_data.py

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now start training with:"
echo "  ./start_training.sh"
echo ""
echo "Or manually:"
echo "  python scripts/train_vetllm.py --data-path processed_data/all_processed_data.json"
echo ""

