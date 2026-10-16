#!/usr/bin/env bash

# Exit on error

echo "🔧 Activating Python venv + CUDA 12.1..."

# Script should be sourced so env vars persist in current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "⚠️  Run with: source scripts/activate_env.sh"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

# Activate venv
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "❌ .venv not found at $VENV_PATH. Create it first with: uv venv --python 3.10"
    return 1 2>/dev/null || exit 1
fi

# Set CUDA environment
if [ -d "/usr/local/cuda-12.1" ]; then
    export CUDA_HOME=/usr/local/cuda-12.1
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=/usr/local/cuda-12.1
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# Force compatible compiler for CUDA 12.1
export CC=gcc-11
export CXX=g++-11

# Optional (helps PyTorch extensions sometimes)
export TORCH_CUDA_ARCH_LIST="8.0"

# Debug info
echo "✅ Python: $(which python)"
echo "✅ CUDA_HOME: $CUDA_HOME"
echo "✅ nvcc: $(which nvcc)"

nvcc --version | grep release || echo "⚠️ nvcc not detected!"

echo "✅ Environment ready 🚀"