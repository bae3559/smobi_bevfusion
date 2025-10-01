# 🚀 BEVFusion + Waymo Training Guide

## ✅ Prerequisites Completed
- WaymoDataset integration ✓
- Flash attention issues fixed ✓
- Docker environment ready ✓
- Configuration files created ✓

## 📋 Training Methods

### Method 1: Direct Training with Working Config (Recommended)

```bash
# Enter Docker container
docker run --gpus all -it --rm -v $(pwd):/workspace bevfusion-waymo bash

# Inside container - set environment
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH

# Option A: Single GPU training
python tools/train_single_gpu.py configs/waymo/bevfusion_waymo_working.yaml \
    --work-dir ./work_dirs/waymo_bevfusion \
    --seed 42

# Option B: If you have multiple GPUs, use distributed training
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

PYTHONPATH=/workspace python tools/train.py configs/waymo/bevfusion_waymo_working.yaml \
    --run-dir ./work_dirs/waymo_bevfusion_dist
```

### Method 2: Copy from Working nuScenes Config

```bash
# 1. Find a working nuScenes config
ls configs/nuscenes/det/

# 2. Copy and modify for Waymo
cp configs/nuscenes/det/centerhead/lssfpn/camera+lidar/default.yaml configs/waymo/bevfusion_from_nuscenes.yaml

# 3. Edit the copied config:
# - Change dataset_type to WaymoDataset
# - Update dataset_root and ann_file paths
# - Adjust object_classes for Waymo
# - Modify point_cloud_range for Waymo

# 4. Train with the modified config
python tools/train_single_gpu.py configs/waymo/bevfusion_from_nuscenes.yaml
```

## 📊 Training Configuration Options

### Basic Configuration (Tested)
```yaml
# Working config: configs/waymo/bevfusion_waymo_working.yaml
- LiDAR only (no camera fusion)
- 6 Waymo object classes
- 397 training samples
- Single GPU friendly
```

### Advanced Configuration
```yaml
# For more powerful setups:
data:
  samples_per_gpu: 2-4  # Increase if you have more GPU memory
  workers_per_gpu: 4-8  # Increase for faster data loading

total_epochs: 50        # More epochs for better performance
```

## 🔧 Training Commands

### Start Training
```bash
# Basic training command
python tools/train_single_gpu.py configs/waymo/bevfusion_waymo_working.yaml \
    --work-dir ./work_dirs/waymo_training

# With specific settings
python tools/train_single_gpu.py configs/waymo/bevfusion_waymo_working.yaml \
    --work-dir ./work_dirs/waymo_training \
    --seed 42 \
    --deterministic
```

### Resume Training
```bash
python tools/train_single_gpu.py configs/waymo/bevfusion_waymo_working.yaml \
    --work-dir ./work_dirs/waymo_training \
    --resume-from ./work_dirs/waymo_training/epoch_10.pth
```

### Validation Only
```bash
python tools/test.py configs/waymo/bevfusion_waymo_working.yaml \
    ./work_dirs/waymo_training/epoch_20.pth \
    --eval bbox
```

## 📈 Monitoring Training

### Check Logs
```bash
# View training logs
tail -f ./work_dirs/waymo_training/*.log

# Check tensorboard (if configured)
tensorboard --logdir ./work_dirs/waymo_training
```

### Expected Output
```
Loading dataset...
✓ Train dataset created with 397 samples
✓ Model created: BEVFusion
✓ GPU ready
Training started...
Epoch 1/20: Loss = X.XX
```

## 🎯 Performance Tips

### Memory Optimization
```yaml
# If running out of GPU memory:
data:
  samples_per_gpu: 1    # Reduce batch size

model:
  heads:
    object:
      num_proposals: 100  # Reduce from 200
```

### Speed Optimization
```yaml
# For faster training:
data:
  workers_per_gpu: 8    # More data loading workers

# Use mixed precision (if supported)
fp16:
  loss_scale: 512.0
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   samples_per_gpu: 1
   ```

2. **Dataset Loading Errors**
   ```bash
   # Check file paths
   ls data/waymo/Waymo_mini/kitti_format/waymo_infos_train.pkl
   ```

3. **Model Configuration Errors**
   ```bash
   # Test config first
   python -c "from mmcv import Config; cfg=Config.fromfile('configs/waymo/bevfusion_waymo_working.yaml'); print('✓ Config OK')"
   ```

### Validation Commands
```bash
# Test dataset loading
python -c "
from mmdet3d.datasets import build_dataset
from mmcv import Config
cfg = Config.fromfile('configs/waymo/bevfusion_waymo_working.yaml')
dataset = build_dataset(cfg.data.train)
print(f'Dataset: {len(dataset)} samples')
"

# Test model creation
python -c "
from mmdet3d.models import build_model
from mmcv import Config
cfg = Config.fromfile('configs/waymo/bevfusion_waymo_working.yaml')
model = build_model(cfg.model)
print(f'Model: {type(model).__name__}')
"
```

## 📁 File Structure
```
bevfusion/
├── configs/waymo/
│   ├── bevfusion_waymo_working.yaml    # Main config
│   ├── final_working.yaml              # Alternative
│   └── simple_bevfusion.yaml           # Simplified
├── data/waymo/Waymo_mini/kitti_format/
│   ├── waymo_infos_train.pkl
│   ├── waymo_infos_val.pkl
│   └── ...
├── work_dirs/waymo_training/           # Training outputs
│   ├── *.log
│   ├── *.pth
│   └── configs.yaml
└── tools/
    ├── train_single_gpu.py
    └── train.py
```

## 🎉 Success Indicators

When everything is working correctly, you should see:
```
✓ Train dataset created with 397 samples
✓ BEVFusion model created
✓ Model moved to GPU successfully
✓ Model parameters: X,XXX,XXX
🎉 Ready to start training!
```

## 📞 Next Steps

1. **Start with basic training** using the working config
2. **Monitor the first few epochs** to ensure stability
3. **Adjust hyperparameters** based on performance
4. **Add camera fusion** once LiDAR-only training is stable
5. **Scale up** to more epochs and larger batch sizes

Your BEVFusion + Waymo integration is ready! 🚀