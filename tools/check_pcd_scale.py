"""
Check if PCD values need scaling.
Based on the first 5 points: X=-0.01, Y=-0.22, Z=-0.20
These look like they are already in meters (20cm range).

But the bbox centers from debug_transform.py were: [11.71, -6.89, 13.14]
which are clearly in meters (car 11m in front, 6m to the side, etc).

So the question is: why are the first 5 merged points still in cm range?
Let me check if LIDAR_TOP_FRONT points are being loaded correctly.
"""

import numpy as np

# Check original merged file
merged_path = 'data/man-truckscenes/samples/LIDAR_TOP_FRONT/LIDAR_TOP_FRONT_1695473372700803_merged.bin'
points = np.fromfile(merged_path, dtype=np.float32).reshape(-1, 5)

print("=== Merged File Analysis ===")
print(f"Total points: {len(points)}")
print()

# The file should contain ~17k points from TOP_FRONT + points from 5 other sensors
# Let's estimate where each sensor's points are

# Assuming points are concatenated in order of sensors
print("=== Checking point ranges in chunks ===")
chunk_size = len(points) // 6  # Approximate

for i in range(6):
    start = i * chunk_size
    end = min((i+1) * chunk_size, len(points))
    chunk = points[start:end]

    print(f"\nChunk {i+1} (points {start} to {end}):")
    print(f"  X: [{chunk[:, 0].min():.2f}, {chunk[:, 0].max():.2f}]")
    print(f"  Y: [{chunk[:, 1].min():.2f}, {chunk[:, 1].max():.2f}]")
    print(f"  Z: [{chunk[:, 2].min():.2f}, {chunk[:, 2].max():.2f}]")
    print(f"  Avg magnitude: {np.linalg.norm(chunk[:, :3], axis=1).mean():.2f}")

print("\n=== Key Observation ===")
print("If first chunk (LIDAR_TOP_FRONT) has values ~0.2m, that's normal sensor frame.")
print("But it should have been kept as-is since it's the reference frame.")
print("Other chunks should be transformed TO this frame.")
print()
print("If later chunks have huge values (100m+), the transformation is wrong.")
