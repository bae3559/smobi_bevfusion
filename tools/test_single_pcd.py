from pypcd4 import PointCloud
import numpy as np

pcd_files = [
    'data/man-truckscenes/samples/LIDAR_TOP_FRONT/LIDAR_TOP_FRONT_1692868171700983.pcd',
]

for pcd_path in pcd_files:
    print(f"\n=== {pcd_path.split('/')[-1]} ===")

    pc = PointCloud.from_path(pcd_path)

    try:
        pc_array = pc.pc_data
    except AttributeError:
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

        # Sample first 5 points
        print("\nFirst 5 points:")
        for i in range(min(5, len(x))):
            print(f"  {i}: x={x[i]:.2f}, y={y[i]:.2f}, z={z[i]:.2f}")
