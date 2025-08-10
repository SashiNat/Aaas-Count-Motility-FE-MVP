# CenterTrack Setup Guide - Azure VM with CUDA 11.7

This README documents the complete setup process for CenterTrack on Azure VM with CUDA 11.7, PyTorch 1.13.1, and DCNv2 via mmcv. This setup resolves all compatibility issues and provides a working training/inference environment. To see what all modifications done to the original CenterTrack repo, visit **MODIFICATIONS.md** file.

## Environment Specifications

- **OS**: Ubuntu 20.04 LTS (Azure VM)
- **System CUDA Driver**: 12.9 (Azure VM pre-installed)
- **CUDA Toolkit**: 11.7 (conda environment - for compilation and PyTorch)
- **PyTorch Version**: 1.13.1+cu117
- **Torchvision Version**: 0.14.1+cu117
- **Python Version**: 3.8
- **DCNv2**: Via mmcv-full 1.7.1 (custom wrapper)

## CUDA Setup Explanation

The Azure VM comes with CUDA 12.9 driver pre-installed, but we use CUDA toolkit 11.7 in our conda environment for better PyTorch compatibility. This works because:

1. **CUDA 12.9 driver is backward compatible** with CUDA 11.7 toolkit
2. **Environment isolation** ensures consistent versions for compilation
3. **PyTorch 1.13.1+cu117** perfectly matches our toolkit version

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n CenterTrack python=3.8 -y
conda activate CenterTrack

# Clean up any existing installations (if needed)
conda clean --all --yes
```

### 2. CUDA 11.7 Installation (Conda Environment)

```bash
# Remove any partial CUDA installation
conda remove cudatoolkit cuda-toolkit --yes

# Install CUDA toolkit 11.7 in conda environment
conda install -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7 --yes

# Set environment variables (make permanent)
mkdir -p ~/miniconda3/envs/CenterTrack/etc/conda/activate.d
mkdir -p ~/miniconda3/envs/CenterTrack/etc/conda/deactivate.d

# Create activation script
cat > ~/miniconda3/envs/CenterTrack/etc/conda/activate.d/cuda.sh << 'EOF'
export CUDA_HOME=~/miniconda3/envs/CenterTrack
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

# Create deactivation script
cat > ~/miniconda3/envs/CenterTrack/etc/conda/deactivate.d/cuda.sh << 'EOF'
unset CUDA_HOME
EOF

# Verify CUDA installation
conda list | grep cuda
nvcc --version  # Should show CUDA 11.7
which nvcc # after setting the path, it should show ~miniconda3/envs/CenterTrack/bin/nvcc
```

### 3. PyTorch Installation

```bash
# Install PyTorch 1.13.1 with CUDA 11.7 support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Verify PyTorch CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 4. CenterTrack Repository Setup (Pre-configured)

```bash
# Clone the pre-configured CenterTrack repository with all fixes applied
cd ~
git clone https://github.com/SashiNat/Aaas-Count-Motility-FE-MVP/tree/main/motility/CenterTrack_Fixed
cd CenterTrack_Fixed

# Install basic requirements
pip install -r requirements.txt

# Install additional required packages
pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install mmcv-full for DCN support
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install headless OpenCV for servers
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### 5. Build External Dependencies

```bash
# Build NMS and other external components
cd ~/CenterTrack_Fixed/src/lib/external
make
```
## Data Preparation

### 1. Data annotation
Data annotation is done in CVAT on Video. Labelling should be done with Bounding box with track consistent over all frames in the video. Once the labelling is done, export the annotations into MOT 1.1 format (which export the data with track-id as well).

```bash
# export cvat annotations
cvat-cli --server-host Azure-VM_external_ip \
    --server-port 8080 \
    --auth cvat_user_name:cvat_password task export-dataset cvat_task_number /path/to/CenterTrack_Fixed/raw_cvat/task_48_export_mot.zip \
    --format "MOT 1.1" \
    --with-images True
```

### 2. Convert data to COCO format

```bash
# unzip the exported mot dataset, which creates two folders gt(classname details, annotations) and img1 (video frames as images)
unzip /path/to/CenterTrack_Fixed/raw_cvat/task_48_export_mot.zip

# covnert cvat MOT 1.1 to COCO format with train/val spilit
python convert_mot_to_coco.py ./raw_cvat/mot/ --output_path ./data/ --train_split 0.8
```
#### **Directory Structure**
After conversion of the dataset to COCO format below is how the directory structure should look like:
```
CenterTrack_Fixed/
├── data/
│   ├── annotations/
│   │   ├── train.json          # COCO format with tracking annotations
│   │   ├── val.json            # Validation annotations  
│   │   └── test.json           # Test annotations (optional)
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   └── test/                   # Test images (optional)
├── models/                     # Pre-trained models
├── exp/                        # Training outputs
└── results/                    # Inference results
```

#### **Annotation Format**
Annotations should be in COCO format with additional tracking fields:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
      "width": 1920,
      "height": 1080,
      "frame_id": 1,        // Frame number in video sequence
      "video_id": 1,        // Video/sequence identifier
      "prev_image_id": -1   // Previous frame ID (-1 for first frame)
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],   // Bounding box
      "area": area,
      "track_id": 1         // Unique track ID across frames
    }
  ],
  "categories": [
    {
      "id": 1, 
      "name": "your_object_class"
    }
  ]
}
```
## Model Training
### 1. Download Pre-trained Models
```bash
# Create models directory
mkdir -p models

# Download MOT17 pre-trained model (example), currently model weights are available publically in google drive. If needed do the required installations.
# Install gdown for Google Drive downloads
pip install gdown
gdown "https://drive.google.com/uc?id=1o_cCo92WiVg8mgwyESd1Gg1AZYnq1iAJ" -O mot17_half_sc.pth

# Or use models from original CenterTrack repository
# Visit: https://github.com/xingyizhou/CenterTrack#models
```

### 2. Basic Training Command
```bash
# Train with custom dataset (hm_disturb, lost_disturb, and fp_disturb are set low at our object size is small)
python src/main.py tracking \
    --arch dla_34 \
    --exp_id mot17_half_sc \
    --dataset custom \
    -custom_dataset_ann_path ./data/annotations/train.json \
    --custom_dataset_img_path ./data/train/ \
    --input_h 384 \
    --input_w 640 \
    --num_classes 1 \
    --batch_size 16 \
    --num_epochs 90 \
    --lr 2.5e-4 \
    --lr_step 30,60 \
    --num_iters 30 \
    --print_iter 10 \
    --load_model ./models/mot17_half_sc.pth \
    --pre_hm \
    --ltrb_amodal \
    --same_aug \
    --aug_rot 0.1 \
    --not_rand_crop \
    --hm_disturb 0.02 \
    --lost_disturb 0.2 \
    --fp_disturb 0.05 \
    --num_workers 0 \
    --save_all \
    --gpus 0
```

### 3. Training Monitoring
```bash
# View training logs
tail -f exp/tracking/my_experiment/logs/train.log

# Check model checkpoints
ls -la exp/tracking/my_experiment/
```

### **Training Output**
Training will save:
- `model_last.pth` - Latest checkpoint
- `model_{epoch}.pth` - Epoch-specific checkpoints
- Training logs and metrics
- Debug visualizations (if enabled)

## Model Evaluation

```bash
# Run evaluation on validation dataset
python src/test.py tracking \
  --arch dla_34 \
  --exp_id mot17_half_sc \
  --dataset custom \
  --custom_dataset_ann_path ./data/annotations/val.json \
  --custom_dataset_img_path ./data/val/ \
  --input_h 384 \
  --input_w 640 \
  --num_classes 1 \
  --load_model ./exp/tracking/mot17_half_sc/model_last.pth \
  --pre_hm \
  --ltrb_amodal \
  --track_thresh 0.4 \
  --pre_thresh 0.5 \
  --debug 2
```

## Model Testing

```bash
# Test on test dataset on Custom Data
python src/demo.py tracking \
    --arch dla_34 \
    --exp_id mot17_half_sc \
    --load_model ./exp/tracking/mot17_half_sc/model_last.pth \
    --input_h 384 \
    --input_w 640 \
    --num_classes 1 \
    --demo ./videos/test.wmv \
    --output_dir ./results \
    --save_video \
    --save_results \
    --match_input_fps \
    --match_input_resolution \
    --pre_hm \
    --debug 0 \
    --no_pause
```
## Jupyter Notebook Setup (Optional)

```bash
# Install Jupyter
conda install jupyter notebook ipykernel -c conda-forge -y

# Add kernel
python -m ipykernel install --user --name CenterTrack --display-name "Python (CenterTrack)"

# Configure for secure remote access
jupyter notebook --generate-config
jupyter notebook password

# Edit config for remote access
cat >> ~/.jupyter/jupyter_notebook_config.py << 'EOF'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = True
c.NotebookApp.allow_root = True
c.NotebookApp.notebook_dir = '/home/azureuser/CenterTrack'
c.NotebookApp.password_required = True
c.NotebookApp.shutdown_no_activity_timeout = 1800
EOF

# Start Jupyter
cd ~/CenterTrack
jupyter notebook
```


## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 11.7 | Conda environment |
| PyTorch | 1.13.1+cu117 | Must match CUDA version |
| Python | 3.8 | Recommended for compatibility |
| mmcv-full | 1.7.1 | For DCN operations |
| torchvision | 0.14.1+cu117 | Must match PyTorch version |

---

**Last Updated**: 10th Aug, 2025.  
**Tested Environment**: Ubuntu 20.04, Azure Standard_NC6s_v3
