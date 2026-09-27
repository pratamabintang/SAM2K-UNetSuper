# SAM2K-UNet: Multi-Modal Landslide Hazard Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1%20%7C%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An advanced multi-modal deep learning architecture for landslide hazard segmentation, integrating **Segment Anything Model 2 (SAM2 Hiera ViT)**, a **ConvNeXt topography encoder**, a **Selective Scan Fusion (SSF / Mamba SSM)** cross-attention module, and a **Kolmogorov-Arnold Network (UKAN / B-spline)** decoder with deep supervision.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Multi-Modal Input Tile    |
                      |   [B, 4, 512, 512] (RGB+DTM)|
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v (Channels 0:3)                v (Channel 3)
        +-------------------------+     +-------------------------+
        |  SAM2 Hiera-Large ViT   |     |      ConvNeXt-Tiny      |
        |      (Optical RGB)      |     |       (Topography)      |
        +------------+------------+     +------------+------------+
                     |                               |
                     +---------------+---------------+
                                     | (Multi-Scale Features)
                                     v
                        +-------------------------+
                        |  SSF Fusion Block (SSM) |
                        | (Selective Scan Mamba)  |
                        +------------+------------+
                                     |
                                     v
                        +-------------------------+
                        |  U-Net Decoder with     |
                        |   UKAN (B-Splines)      |
                        +------------+------------+
                                     |
                     +---------------+---------------+
                     |               |               |
                     v (side2)       v (side1)       v (head)
                Prediction 2    Prediction 1     Final Prediction
```

- **Dual-Branch Encoder**: Optical RGB features are extracted via SAM2 Hiera-L, while physical elevation (DTM) is processed through a dedicated ConvNeXt-Tiny CNN.
- **Selective Scan Fusion (SSF)**: 2D State Space Model (SSM / Mamba) cross-directionally fuses optical spatial features with continuous topography.
- **KAN Decoder**: Learnable non-linear univariate B-spline activation functions on feature edges replace standard MLP layers for boundary delineation.
- **Deep Supervision**: Auxiliary loss computed across 3 multi-scale decoder stages via `structure_loss` (Weighted BCE + Weighted IoU).

---

## Table of Contents
1. [Prerequisites & Environment](#prerequisites--environment)
2. [Step-by-Step Setup Guide (Google Colab)](#step-by-step-setup-guide-google-colab)
3. [Dataset Directory Layout](#dataset-directory-layout)
4. [Training](#training)
5. [Inference & Testing](#inference--testing)
6. [Evaluation (SOD Metrics)](#evaluation-sod-metrics)
7. [Visualizations](#visualizations)
8. [Numerical Stability & Troubleshooting](#numerical-stability--troubleshooting)

---

## Prerequisites & Environment

- **Operating System**: Linux (Ubuntu 20.04 / 22.04 / Debian) or Windows
- **Python**: 3.10 – 3.13
- **PyTorch**: $\ge 2.0.0$ (Verified on PyTorch 2.11.0+cu128 with CUDA 12.8)
- **Supported Accelerators**: 
  - NVIDIA Blackwell (SM120 / CC 12.0, e.g., RTX PRO 6000 Server Edition)
  - NVIDIA Hopper / Ada Lovelace / Ampere (SM80–SM90, e.g., A100, H100, RTX 4090)
  - NVIDIA Turing / Volta (SM70–SM75, e.g., T4, V100, RTX 2060)

---

## Step-by-Step Setup Guide (Google Colab)

Open a new Google Colab notebook and set the hardware accelerator to **GPU** (`Runtime` $\to$ `Change runtime type` $\to$ `T4`, `A100`, or `Blackwell/L4`).

### Step 1: Mount Google Drive (Optional but Recommended)
If your dataset or checkpoints are stored in Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone or Upload the Repository
```bash
# Clone your repository or upload the project folder
%cd /content
!git clone https://github.com/pratamabintang/SAM2K-UNetSuper.git
%cd /content/SAM2K-UNetSuper
```

### Step 3: Install Required Python Packages
```bash
!pip install -r requirements.txt
```

### Step 4: Compile the Custom CUDA Selective Scan Extension
The custom CUDA C++ CORE backend accelerates 2D selective scan computations. Compile it directly in your environment:

```bash
%cd /content/SAM2K-UNetSuper/selective_scan

# Clean any existing artifacts
!rm -rf build/ dist/ *.egg-info *.so

# (Optional) Export GPU compute capability (e.g. 12.0 for Blackwell, 8.0 for A100, 7.5 for T4)
!export TORCH_CUDA_ARCH_LIST="auto"

# Compile and install in editable mode
!pip install --no-build-isolation -e .

# Test that the extension imports cleanly (always import torch first to load libc10.so)
!python -c "import torch; import selective_scan_cuda_core; print('SUCCESS: selective_scan_cuda_core loaded!')"

%cd /content/SAM2K-UNetSuper
```

### Step 5: Download SAM2 Pretrained Weights
Download the official SAM2 Hiera-Large checkpoint into `checkpoints/`:
```bash
!mkdir -p checkpoints
!wget -O checkpoints/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

---

## Dataset Directory Layout

The repository expects multi-modal rasters organized into `train`, `val`, and `test` splits. Each split contains aligned raster directories:

```
datasets/landslide/
├── black_list.txt            # IDs of corrupted/noisy samples to exclude
├── train/
│   ├── IMAGE/                # Optical RGB (.png or .tif)
│   ├── DTM/                  # Digital Terrain Model (.tif or .png)
│   └── LABEL/                # Ground truth binary mask (.png or .tif)
├── val/
│   ├── IMAGE/
│   ├── DTM/
│   └── LABEL/
└── test/
    ├── IMAGE/
    ├── DTM/
    └── LABEL/
```

> **Note on Data Staging in Colab**: Reading thousands of small TIFF files directly from Google Drive (`/content/drive/MyDrive/...`) can experience I/O throttling. For fastest training, stage the rasters to Colab local disk (`/content/data/`) using `rsync` or `unzip`.

---

## Training

### Method A: Automated Bash Pipeline (Recommended for Colab)
The `train.sh` script automates DTM ablation experiments (Control 0% with synthetic zero DTM vs. 100% with true DTM) while staging data safely:

1. Open `train.sh` and set your paths:
   ```bash
   OUTPUT_TRAINING_PATH="/content/drive/MyDrive/project/runs"
   HIERA_PATH="/content/SAM2K-UNetSuper/checkpoints/sam2_hiera_large.pt"
   TRAIN_IMAGE_PATH="/content/data/train/IMAGE"
   TRAIN_MASK_PATH="/content/data/train/LABEL"
   TRAIN_DTM_PATH="/content/data/train/DTM"
   VAL_IMAGE_PATH="/content/data/val/IMAGE"
   VAL_MASK_PATH="/content/data/val/LABEL"
   VAL_DTM_PATH="/content/data/val/DTM"
   ```

2. Execute training:
   ```bash
   !bash train.sh
   ```

### Method B: Direct Python Execution
You can also launch training directly using YAML configurations:

```bash
!python train.py --config configs/default.yaml
```

**Key CLI overrides**:
```bash
!python train.py \
    --config configs/default.yaml \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0001
```

**Configuration Settings (`configs/default.yaml`)**:
```yaml
experiment:
  name: "base"
  seed: 42
  runs_dir: "runs"

dataset:
  data_dir: "datasets/landslide"
  modalities:
    - "IMAGE"
    - "DTM"
  blacklist_path: "datasets/landslide/black_list.txt"
  size: 512
  batch_size: 4
  num_workers: 4

model:
  hiera_path: "checkpoints/sam2_hiera_large.pt"
  topo_backbone: "convnext_tiny"
  pretrained_topo: true
  use_kan: true
  use_ssf: true

training:
  epochs: 50
  lr: 0.0001
  weight_decay: 0.0005
  min_lr: 1.0e-7
  amp: true
  amp_dtype: "auto"       # Automatically selects bfloat16 on SM80+/SM120
  grad_clip: 1.0          # Norm clipping threshold
  save_interval: 5
  eval_interval: 1
```

---

## Inference & Testing

To generate probability maps and binary prediction masks on the test split:

```bash
!python test.py \
    --config configs/default.yaml \
    --checkpoint runs/base/checkpoints/best.pth \
    --split test \
    --save_dir "results/test_predictions" \
    --save_masks true \
    --threshold 0.5 \
    --batch_size 1
```

Or using `test.sh`:
```bash
!bash test.sh
```

Outputs will be saved in `--save_dir`:
- `probability_maps/`: Continuous sigmoid confidence rasters ($[0.0, 1.0]$)
- `binary_masks/`: Thresholded binary masks ($0$ or $255$)
- `metrics.json`: Per-sample and overall test metrics (IoU, Dice, Precision, Recall, MAE, mAP)

---

## Evaluation (SOD Metrics)

To compute standard Salient Object Detection (SOD) benchmark metrics including F-measure ($F_\beta$), Weighted F-measure ($F_\beta^w$), S-measure ($S_m$), E-measure ($E_m$), and MAE:

```bash
!python eval.py \
    --dataset_name "landslide" \
    --pred_path "results/test_predictions/probability_maps" \
    --gt_path "datasets/landslide/test/LABEL"
```

---

## Visualizations

To render 4-panel comparison figures (**Optical Image**, **Ground Truth Label**, **Model Prediction**, and **Confidence Heatmap**):

### Mode 1: From Existing Test Outputs (Fast, No GPU Load)
```bash
!python visualize.py \
    --results_dir "results/test_predictions" \
    --data_dir "datasets/landslide" \
    --split test \
    --output_dir "results/visualizations" \
    --num_samples 10 \
    --threshold 0.5
```

### Mode 2: Direct End-to-End Checkpoint Inference
```bash
!python visualize.py \
    --checkpoint "runs/base/checkpoints/best.pth" \
    --config "configs/default.yaml" \
    --split test \
    --output_dir "results/visualizations" \
    --num_samples 10 \
    --threshold 0.5
```

---

## Numerical Stability & Troubleshooting

If you observe intermittent `Batch Loss: NaN` or `Validation Loss: NaN`:

1. **Mixed Precision Policy (`amp_dtype: auto`)**:
   - On NVIDIA Blackwell (SM120) or Ampere/Hopper (SM80+), `torch.bfloat16` is used automatically. BFloat16 has an 8-bit exponent ($10^{38}$ range), completely avoiding the half-precision overflow limit ($65,504$) in long sequence selective scans.
   - For older GPUs (T4 / Turing), FP16 is used with automatic FP32 input casting via `SelectiveScan.apply`.
2. **Selective Scan FP32 Promotion**:
   - Both `DBISSF_Attention` and `CM_Attention` in `fusion.py` wrap calls to `selective_scan_cuda` with `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)`.
3. **Loss Function Guards**:
   - `structure_loss` explicitly promotes predictions and masks to `torch.float32` prior to spatial pooling and adds `+ 1e-8` to the weight reduction denominator.
4. **Dataset Blacklist**:
   - Ensure `blacklist_path: "datasets/landslide/black_list.txt"` is specified in your config to filter out known corrupted or uncalibrated tiles.
5. **`ImportError: libc10.so: cannot open shared object file`**:
   - `libc10.so` is an internal PyTorch library residing in `torch/lib`. In Python scripts, always `import torch` before importing `selective_scan_cuda_core`. The updated `selective_scan/setup.py` also automatically embeds `-Wl,-rpath,<torch/lib>` into the compiled binary on Linux.

---

## Citation & Architecture Records
For detailed technical analyses and architectural design rationales, refer to:
- [docs/nan_loss_diagnosis.md](docs/nan_loss_diagnosis.md): In-depth numerical diagnosis and Blackwell CUDA extension analysis.
- [docs/adr/0001-per-tile-dtm-normalization.md](docs/adr/0001-per-tile-dtm-normalization.md): Decision record for local tile relief normalization.
- [docs/adr/0002-dual-branch-topography-encoder.md](docs/adr/0002-dual-branch-topography-encoder.md): Asymmetric dual-branch encoder architecture.
- [docs/adr/0003-selective-scan-precision-and-amp-stabilization.md](docs/adr/0003-selective-scan-precision-and-amp-stabilization.md): Selective scan precision policy and mixed-precision stabilization.
