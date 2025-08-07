# CenterTrack-Fixed

Modified CenterTrack implementation with CUDA 11.7 and PyTorch 1.13.1 compatibility fixes for Azure VM.

## Quick Setup

### Prerequisites
- Azure VM with GPU support
- CUDA 12.9 driver (pre-installed on Azure)

### 1. Environment Setup
conda create -n CenterTrack python=3.8 -y
conda activate CenterTrack

# Install CUDA toolkit 11.7
conda install -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7 --yes

# Set environment variables (permanent)
mkdir -p ~/miniconda3/envs/CenterTrack/etc/conda/activate.d
mkdir -p ~/miniconda3/envs/CenterTrack/etc/conda/deactivate.d

cat > ~/miniconda3/envs/CenterTrack/etc/conda/activate.d/cuda.sh << 'CUDA_EOF'
export CUDA_HOME=~/miniconda3/envs/CenterTrack
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
CUDA_EOF

cat > ~/miniconda3/envs/CenterTrack/etc/conda/deactivate.d/cuda.sh << 'CUDA_EOF'
unset CUDA_HOME
CUDA_EOF

# Verify installation
nvcc --version  # Should show CUDA 11.7

### 3. Install Dependencies
# Navigate to this CenterTrack-Fixed directory
cd motility/CenterTrack-Fixed

# Install PyTorch with CUDA 11.7 support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other requirements
pip install -r requirements.txt
pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Build external dependencies
cd src/lib/external && make && cd ../../..

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"



Key Modifications Applied
This repository contains a pre-configured CenterTrack with the following fixes:
1. Custom DCNv2 Implementation (src/lib/model/networks/DCNv2/)

Problem: Original DCNv2 incompatible with PyTorch 1.13+
Solution: Custom wrapper using mmcv DeformConv2dPack
Features:

Handles both deform_groups and deformable_groups parameters
Manual bias handling (mmcv limitation workaround)
Single input forward method matching original interface



2. Torchvision Compatibility (src/lib/model/networks/backbones/mobilenet.py)

Problem: Import path changed in newer torchvision versions
Solution: Updated import from torchvision.models.utils to torch.hub

3. Headless Demo Script (src/demo_headless.py)

Problem: Original demo requires display (cv2.imshow) - not available on Azure VM
Solution: Custom headless demo script with:

No GUI dependencies
Proper video processing and error handling
JSON results output



4. Model Files Excluded

Problem: Model files are 77MB-227MB each
Solution: Excluded from git, documented in README files

### Training
```bash
python src/main.py tracking \
  --exp_id your_experiment \
  --dataset custom \
  --custom_dataset_ann_path ./data/annotations/train.json \
  --custom_dataset_img_path ./data/train/ \
  --load_model ./models/pretrained_model.pth
```

### Inference (Headless)
```bash
# Set environment for headless operation
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# Run demo
python src/demo_headless.py tracking \
  --load_model ./exp/tracking/your_experiment/model_last.pth \
  --demo ./videos/your_video.mp4 \
  --exp_id demo_test
```
Original Repository
Based on: https://github.com/xingyizhou/CenterTrack
License
See original CenterTrack license.

