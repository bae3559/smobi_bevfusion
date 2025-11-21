import numpy as np
import pickle

# Load mantruck data
with open('data/man-truckscenes/mantruck_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]

print("=== Sample Info ===")
print(f"LiDAR path: {sample['lidar_path']}")
print()

# Load merged LiDAR
if sample['lidar_path'].endswith('_merged.bin'):
    points = np.fromfile(sample['lidar_path'], dtype=np.float32).reshape(-1, 5)

    print("=== Merged LiDAR Full Statistics ===")
    print(f"Total points: {len(points)}")
    print(f"Shape: {points.shape}")
    print()

    for i, name in enumerate(['X', 'Y', 'Z', 'Intensity', 'Timestamp']):
        data_col = points[:, i]
        print(f"{name}:")
        print(f"  Range: [{data_col.min():.2f}, {data_col.max():.2f}]")
        print(f"  Mean: {data_col.mean():.2f}")
        print(f"  Std: {data_col.std():.2f}")
        if name in ['X', 'Y', 'Z']:
            print(f"  Positive: {(data_col > 0).sum()} ({(data_col > 0).sum()/len(points)*100:.1f}%)")
            print(f"  Negative: {(data_col < 0).sum()} ({(data_col < 0).sum()/len(points)*100:.1f}%)")
        print()

    # Check for outliers
    print("=== Outlier Analysis ===")
    for i, name in enumerate(['X', 'Y', 'Z']):
        data_col = points[:, i]
        q1, q3 = np.percentile(data_col, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outliers = (data_col < lower_bound) | (data_col > upper_bound)
        print(f"{name} outliers (beyond 3*IQR): {outliers.sum()} ({outliers.sum()/len(points)*100:.2f}%)")
    print()

    # Sample some points
    print("=== Sample Points ===")
    print("First 5 points:")
    for i in range(min(5, len(points))):
        print(f"  {i}: X={points[i,0]:.2f}, Y={points[i,1]:.2f}, Z={points[i,2]:.2f}, I={points[i,3]:.2f}")
    print()
    print("Last 5 points:")
    for i in range(max(0, len(points)-5), len(points)):
        print(f"  {i}: X={points[i,0]:.2f}, Y={points[i,1]:.2f}, Z={points[i,2]:.2f}, I={points[i,3]:.2f}")

else:
    print("ERROR: Not a merged file!")
