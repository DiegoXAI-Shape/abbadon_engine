#!/bin/bash
# Compile, run tests and, only if tests pass, run the engine with the provided image.
# Use: ./run.sh <image>
#   ./run.sh images/fredo.jpeg <-- My dog's image

# if ANY command fails (exit code != 0), the script stops there
set -e

# Like `if (argc < 2)` of engine.cpp: $# = how many arguments, $0 = script's name
if [ $# -lt 1 ]; then
    echo "Using: $0 <image>"
    exit 1
fi

# Conda includes it's own compiler and nvcc 13.3 → breaks CMake's configuration
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "[Warning] Conda is active ($CONDA_DEFAULT_ENV). Run: 'conda deactivate' before run setup.sh"
    exit 1
fi

# Always work from the root of the project. (the location where this script is located),
# because the paths weights/ and images/ are relatives
cd "$(dirname "$0")"

# Download the weights from Hugging Face only if they are missing
HF_URL="https://huggingface.co/DiegoXAI-Shape/abbadon-engine/resolve/main"
mkdir -p weights
for f in Daowa_Oracle_Frozen.pt mendicant_bias_cpp.pt; do
    if [ ! -f "weights/$f" ]; then
        echo "==> Downloading weights/$f from Hugging Face..."
        # -f: fail on HTTP errors instead of saving an error page as the .pt
        curl -fL -o "weights/$f" "$HF_URL/$f"
    fi
done

# Configure only if don't exist build/
if [ ! -d build ]; then
    echo "==> Configuring CMake (first time)..."
    cmake -S . -B build \
        -DCMAKE_PREFIX_PATH="$(pwd)/env_engine/lib/python3.12/site-packages/torch/share/cmake/Torch" \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DTORCH_CUDA_ARCH_LIST="12.0"
fi

echo "==> Compiling..."
cmake --build build -j

echo "==> Running Tests..."
./build/engine_tests
# If some test fails, engine_tests returns 1 and `set -e` stops all here

echo "==> Tests OK. Running engine with $1..."
./build/engine "$1"
