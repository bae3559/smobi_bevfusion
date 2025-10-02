# BEVFusion Training Guide

This repository contains BEVFusion implementation for both **Waymo** and **nuScenes** datasets with complete dataset preparation, training, testing, and visualization pipelines.

## 🚀 Quick Start

### Requirements
- Python 3.8+
- CUDA 11.1+
- PyTorch 1.9+

### Installation
```bash
# Clone repository
git clone https://github.com/bae3559/smobi_bevfusion.git
cd bevfusion

# Install dependencies
#pip install -r requirements.txt
python setup.py develop
```

## 📊 Dataset Preparation

### 1. Waymo Dataset

#### Download Data
```bash
# Download from Waymo Open Dataset
# Place raw data in: data/waymo/raw/
```

#### Convert to BEVFusion Format
```bash
# Convert Waymo data (full dataset)
python tools/create_data.py waymo \
    --root-path data/waymo/ \
    --out-dir data/waymo/Waymo_processed \
    --extra-tag waymo \
    --version full

# Convert Waymo mini dataset (for testing)
python tools/create_data.py waymo \
    --root-path data/waymo/Waymo_mini \
    --out-dir data/waymo/Waymo_processed \
    --extra-tag waymo \
    --version mini
```

#### Expected Directory Structure
```
data/waymo/
├── Waymo_processed/
│   ├── waymo_infos_train.pkl
│   ├── waymo_infos_val.pkl
│   ├── waymo_infos_test.pkl
│   └── waymo_dbinfos_train.pkl
└── Waymo_mini/
    ├── training/
    ├── validation/
    └── testing/
```

### 2. nuScenes Dataset

#### Download Data
```bash
# Download from nuScenes website
# Place data in: data/nuscenes/
```

#### Convert to BEVFusion Format
```bash
# Convert nuScenes data
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
```

#### Expected Directory Structure
```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_dbinfos_train.pkl
```

## 🏋️ Training

### Waymo Training

#### 1. Single GPU Training
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/waymo/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
```

#### 2. Multi-GPU Training
```bash
torchpack dist-run -np 8 python tools/train.py \
    configs/waymo/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
```

#### 3. Configuration Options
- **Camera + LiDAR**: `configs/waymo/det/transfusion/secfpn/camera+lidar/default.yaml`
- **LiDAR Only**: `configs/waymo/det/transfusion/secfpn/lidar/default.yaml`
- **Camera Only**: `configs/waymo/det/centerhead/lssfpn/camera/default.yaml`

### nuScenes Training

#### 1. Single GPU Training
```bash
torchpack dist-run -np 1 python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth 
```

#### 2. Multi-GPU Training
```bash
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --load_from pretrained/lidar-only-det.pth 
```

#### 3. Configuration Options
- **Camera + LiDAR**: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`
- **LiDAR Only**: `configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`
- **Camera Only**: `configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml`

### Training Options
```bash
# Resume from checkpoint
python tools/train.py CONFIG --resume-from CHECKPOINT

# Load pretrained weights
python tools/train.py CONFIG --load-from PRETRAINED

# Specify GPU IDs
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py CONFIG
```

## 🧪 Testing

### Waymo Testing

```bash
# Test with trained model
torchpack dist-run -np 1 python tools/test.py \
    runs/run-5e3259ec/configs.yaml \
    runs/run-5e3259ec/epoch_20.pth \
    --eval bbox

# Multi-GPU testing
torchpack dist-run -np 8 python tools/test.py \
    runs/run-5e3259ec/configs.yaml \
    runs/run-5e3259ec/epoch_20.pth \
    --eval bbox
```

#### Expected Waymo Results
```
Waymo Evaluation Results:
mAP: 0.3588
mATE: 0.4818
mASE: 0.5229
mAOE: 0.6159
mAVE: 0.0000
mAAE: 0.0000
NDS: 0.4308

Per-class results:
Object Class              AP        ATE       ASE       AOE       AVE       AAE
vehicle                   0.628     0.376     0.313     0.536     0.000     0.000
pedestrian                0.561     0.134     0.387     0.349     0.000     0.000
cyclist                   0.000     1.000     1.000     1.000     0.000     0.000
sign                      0.577     0.166     0.195     0.345     0.000     0.000
```

### nuScenes Testing

```bash
# Test with trained model
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    checkpoints/nuscenes_model.pth \
    --eval bbox

# Multi-GPU testing
torchpack dist-run -np 8 python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    checkpoints/nuscenes_model.pth \
    --eval bbox
```

#### Expected nuScenes Results
```
mAP: 0.3588
mATE: 0.4818
mASE: 0.5229
mAOE: 0.6159
mAVE: 0.5402
mAAE: 0.3253
NDS: 0.4308

Per-class results:
Object Class            AP        ATE       ASE       AOE       AVE       AAE
car                     0.318     0.376     0.313     0.536     0.248     0.202
truck                   0.194     0.430     0.264     0.300     0.131     0.006
bus                     0.947     0.204     0.127     0.063     0.849     0.271
trailer                 0.000     1.000     1.000     1.000     1.000     1.000
construction_vehicle    0.000     1.000     1.000     1.000     1.000     1.000
pedestrian              0.805     0.134     0.387     0.349     0.241     0.120
motorcycle              0.039     0.421     0.466     0.950     0.061     0.004
bicycle                 0.467     0.166     0.195     0.345     0.791     0.000
traffic_cone            0.819     0.087     0.476     nan       nan       nan
barrier                 0.000     1.000     1.000     1.000     nan       nan
```

## 👁️ Visualization (코드 수정 필요)

### Waymo Visualization

#### Ground Truth Visualization
```bash
# Visualize GT boxes
python tools/visualize.py \
    configs/waymo/det/transfusion/secfpn/camera+lidar/default.yaml \
    --mode gt \
    --split val \
    --out-dir viz_waymo_gt

# Visualize specific classes
python tools/visualize.py \
    configs/waymo/det/transfusion/secfpn/camera+lidar/default.yaml \
    --mode gt \
    --split val \
    --bbox-classes 0 1 \
    --out-dir viz_waymo_vehicle_pedestrian
```

#### Prediction Visualization
```bash
# Visualize model predictions
python tools/visualize.py \
    configs/waymo/det/transfusion/secfpn/camera+lidar/default.yaml \
    --mode pred \
    --checkpoint checkpoints/waymo_model.pth \
    --split val \
    --bbox-score 0.3 \
    --out-dir viz_waymo_pred
```

#### Class Mapping (Waymo)
- `0`: vehicle
- `1`: pedestrian
- `2`: cyclist
- `3`: sign

### nuScenes Visualization

#### Ground Truth Visualization
```bash
# Visualize GT boxes
python tools/visualize.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --mode gt \
    --split val \
    --out-dir viz_nuscenes_gt

# Visualize specific classes
python tools/visualize.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --mode gt \
    --split val \
    --bbox-classes 0 1 2 \
    --out-dir viz_nuscenes_car_truck_bus
```

#### Prediction Visualization
```bash
# Visualize model predictions
python tools/visualize.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --mode pred \
    --checkpoint checkpoints/nuscenes_model.pth \
    --split val \
    --bbox-score 0.3 \
    --out-dir viz_nuscenes_pred
```

#### Class Mapping (nuScenes)
- `0`: car
- `1`: truck
- `2`: construction_vehicle
- `3`: bus
- `4`: trailer
- `5`: barrier
- `6`: motorcycle
- `7`: bicycle
- `8`: pedestrian
- `9`: traffic_cone

