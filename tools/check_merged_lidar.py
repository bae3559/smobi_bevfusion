import pickle
import numpy as np

# Load mantruck data
with open('data/man-truckscenes/mantruck_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]

print("=== LiDAR path ===")
print(sample['lidar_path'])
print()

# Load merged LiDAR
if sample['lidar_path'].endswith('_merged.bin'):
    points = np.fromfile(sample['lidar_path'], dtype=np.float32).reshape(-1, 5)

    print("=== Merged LiDAR Statistics ===")
    print(f"Total points: {len(points)}")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print()

    # Check point distribution
    print("=== Point Distribution ===")
    print(f"Points with X > 0: {(points[:, 0] > 0).sum()} ({(points[:, 0] > 0).sum()/len(points)*100:.1f}%)")
    print(f"Points with X < 0: {(points[:, 0] < 0).sum()} ({(points[:, 0] < 0).sum()/len(points)*100:.1f}%)")
    print(f"Points with Y > 0: {(points[:, 1] > 0).sum()} ({(points[:, 1] > 0).sum()/len(points)*100:.1f}%)")
    print(f"Points with Y < 0: {(points[:, 1] < 0).sum()} ({(points[:, 1] < 0).sum()/len(points)*100:.1f}%)")
    print()

    # Check if points are beyond typical range
    beyond_50 = (np.abs(points[:, 0]) > 50) | (np.abs(points[:, 1]) > 50)
    print(f"Points beyond ±50m: {beyond_50.sum()} ({beyond_50.sum()/len(points)*100:.1f}%)")

else:
    print("ERROR: Not a merged file!")
