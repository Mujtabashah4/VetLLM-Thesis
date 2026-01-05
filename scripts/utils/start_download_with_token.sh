#!/bin/bash
# Start download with token automatically loaded

cd "$(dirname "$0")"

# Load token from file
TOKEN_FILE="$HOME/.cache/huggingface/token"
if [ -f "$TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$TOKEN_FILE")
    export HUGGINGFACE_HUB_TOKEN=$(cat "$TOKEN_FILE")
    echo "✅ Token loaded from: $TOKEN_FILE"
else
    echo "⚠️  Token file not found at: $TOKEN_FILE"
    echo "   Trying to continue anyway..."
fi

# Start download
echo "🚀 Starting download..."
python download_llama3.1.py

