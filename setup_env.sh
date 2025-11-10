#!/bin/bash

# ============================================
# Setup script for MLSP Project
# Hardware: 9950X3D + RTX 5090 + 64GB RAM
# CUDA: 13.0
# ============================================

echo "Setting up MLSP Project environment..."

# Create conda environment
echo "Creating conda environment: mlsp_project"
conda create -n mlsp_project python=3.11 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlsp_project

# Install PyTorch with CUDA 13.0 support
echo "Installing PyTorch with CUDA 13.0..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install system dependencies for PyAudio
echo "Installing system dependencies for audio processing..."
echo "You may need to run: sudo apt-get install portaudio19-dev python3-pyaudio"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Setup complete! Activate environment with: conda activate mlsp_project"
