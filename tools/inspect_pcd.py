from pypcd4 import PointCloud
import numpy as np

# Load a sample PCD file
pcd_path = 'data/man-truckscenes/samples/LIDAR_TOP_FRONT/LIDAR_TOP_FRONT_1695473372700803.pcd'

print(f"Loading: {pcd_path}")
pc = PointCloud.from_path(pcd_path)

# Get data
try:
    pc_array = pc.pc_data
except AttributeError:
    try:
        pc_array = pc.numpy()
    except:
        pc_array = pc.to_ndarray()

print("\n=== PCD Structure ===")
print(f"Shape: {pc_array.shape}")
print(f"Dtype: {pc_array.dtype}")

if pc_array.dtype.names:
    print(f"\nFields: {pc_array.dtype.names}")
    print("\n=== Field Statistics ===")
    for field in pc_array.dtype.names:
        data = pc_array[field]
        print(f"{field}:")
        print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Std: {data.std():.2f}")
else:
    print("\nNo structured array - regular array")
    print(f"Shape: {pc_array.shape}")

# Try extracting xyz
print("\n=== Trying to extract XYZ ===")
if pc_array.dtype.names:
    if 'x' in pc_array.dtype.names:
        x = pc_array['x']
        y = pc_array['y']
        z = pc_array['z']
        print(f"X: [{x.min():.2f}, {x.max():.2f}]")
        print(f"Y: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Z: [{z.min():.2f}, {z.max():.2f}]")
