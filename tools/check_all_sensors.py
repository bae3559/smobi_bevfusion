from pypcd4 import PointCloud
import numpy as np
import pickle

# Load mantruck data
with open('data/man-truckscenes/mantruck_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]

# Get sensor file paths from sample
sensors_info = {
    'LIDAR_TOP_FRONT': 'samples/LIDAR_TOP_FRONT/LIDAR_TOP_FRONT_1692868171700983.pcd',
    'LIDAR_TOP_LEFT': 'samples/LIDAR_TOP_LEFT/LIDAR_TOP_LEFT_1692868171700983.pcd',
    'LIDAR_TOP_RIGHT': 'samples/LIDAR_TOP_RIGHT/LIDAR_TOP_RIGHT_1692868171700983.pcd',
    'LIDAR_LEFT': 'samples/LIDAR_LEFT/LIDAR_LEFT_1692868171700983.pcd',
    'LIDAR_RIGHT': 'samples/LIDAR_RIGHT/LIDAR_RIGHT_1692868171700983.pcd',
    'LIDAR_REAR': 'samples/LIDAR_REAR/LIDAR_REAR_1692868171700983.pcd',
}

for sensor_name, rel_path in sensors_info.items():
    path = f'data/man-truckscenes/{rel_path}'

    print(f"\n=== {sensor_name} ===")
    print(f"Path: {path}")

    # Load PCD
    pc = PointCloud.from_path(str(path))
    try:
        pc_array = pc.pc_data
    except:
        try:
            pc_array = pc.numpy()
        except:
            pc_array = pc.to_ndarray()

    if pc_array.dtype.names:
        x = pc_array['x']
        y = pc_array['y']
        z = pc_array['z']

        print(f"Points: {len(x)}")
        print(f"X: [{x.min():.2f}, {x.max():.2f}]")
        print(f"Y: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Z: [{z.min():.2f}, {z.max():.2f}]")

        # First 3 points
        print("First 3 points:")
        for i in range(min(3, len(x))):
            print(f"  {i}: x={x[i]:.2f}, y={y[i]:.2f}, z={z[i]:.2f}")
