from truckscenes.truckscenes import TruckScenes
from pyquaternion import Quaternion
from pypcd4 import PointCloud
import numpy as np

nusc = TruckScenes(version='v1.0-mini', dataroot='data/man-truckscenes/', verbose=False)

# Get first sample
sample = nusc.sample[0]
print(f"Sample token: {sample['token'][:8]}")

# Get LiDAR info
lidar_token = sample['data']['LIDAR_TOP_FRONT']
sd_rec = nusc.get('sample_data', lidar_token)
cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])

print("\n=== Transformations ===")
print(f"lidar2ego translation: {cs_rec['translation']}")
print(f"lidar2ego rotation (quat): {cs_rec['rotation']}")
print(f"ego2global translation: {pose_rec['translation']}")
print(f"ego2global rotation (quat): {pose_rec['rotation']}")

# Load PCD directly
lidar_path = nusc.get_sample_data_path(lidar_token)
print(f"\n=== Loading PCD: {lidar_path} ===")

pc = PointCloud.from_path(str(lidar_path))
try:
    pc_array = pc.pc_data
except:
    try:
        pc_array = pc.numpy()
    except:
        pc_array = pc.to_ndarray()

if pc_array.dtype.names:
    x = pc_array['x'][:100]  # First 100 points
    y = pc_array['y'][:100]
    z = pc_array['z'][:100]

    print(f"\nFirst 5 points (raw from PCD):")
    for i in range(5):
        print(f"  {i}: x={x[i]:.2f}, y={y[i]:.2f}, z={z[i]:.2f}")

    print(f"\nStatistics (first 100 points):")
    print(f"X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Y: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Z: [{z.min():.2f}, {z.max():.2f}]")

# Also check what get_sample_data returns
print("\n=== Using nuScenes API ===")
_, boxes, _ = nusc.get_sample_data(lidar_token)
print(f"Number of boxes: {len(boxes)}")
if len(boxes) > 0:
    print(f"First box center: {boxes[0].center}")
