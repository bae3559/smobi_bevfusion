#!/usr/bin/env python3

import numpy as np

def test_projection():
    """Test camera projection with typical Waymo values"""

    # Typical Waymo camera intrinsics (based on converter code)
    fx, fy = 2050.0, 2050.0  # focal lengths
    cx, cy = 960.0, 640.0    # principal point

    print("=== Camera Projection Test ===")
    print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # Camera intrinsic matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    print(f"Intrinsic matrix K:")
    print(K)

    # Typical image size (Waymo)
    img_width, img_height = 1920, 1280
    print(f"Image size: {img_width}x{img_height}")

    # Test points in camera coordinate system
    print(f"\n=== Test Points in Camera Coordinates ===")

    # Test points at different distances
    test_points_cam = np.array([
        [0, 0, 5],     # 5m ahead, center
        [1, 0, 5],     # 5m ahead, 1m right
        [0, 1, 5],     # 5m ahead, 1m up
        [2, 1, 10],    # 10m ahead, 2m right, 1m up
        [-2, -1, 15],  # 15m ahead, 2m left, 1m down
    ])

    for i, pt in enumerate(test_points_cam):
        print(f"Point {i}: camera_coords={pt}")

        # Project to image
        x_cam, y_cam, z_cam = pt
        if z_cam > 0:  # Valid depth
            u = fx * x_cam / z_cam + cx
            v = fy * y_cam / z_cam + cy
            print(f"  -> image_coords=({u:.1f}, {v:.1f})")

            # Check if in bounds
            in_bounds = 0 <= u < img_width and 0 <= v < img_height
            print(f"  -> in_bounds={in_bounds}")
        else:
            print(f"  -> invalid (z={z_cam} <= 0)")
        print()

    # Test what happens with very large coordinates
    print(f"=== Test Large Coordinates ===")
    large_point = np.array([100, 50, 5])  # 100m right, 50m up, 5m ahead
    x, y, z = large_point
    u = fx * x / z + cx
    v = fy * y / z + cy
    print(f"Large point: {large_point} -> image_coords=({u:.1f}, {v:.1f})")
    print(f"Way outside image bounds (expected)")

    # Test lidar2camera transformation effect
    print(f"\n=== Test LiDAR to Camera Transformation ===")

    # Typical lidar2camera matrix (example)
    # LiDAR: x-forward, y-left, z-up
    # Camera: x-right, y-down, z-forward
    lidar2camera = np.array([
        [ 0,  0,  1,  0],   # camera_x = lidar_z
        [-1,  0,  0,  0],   # camera_y = -lidar_x
        [ 0, -1,  0,  2],   # camera_z = -lidar_y + 2 (camera height)
        [ 0,  0,  0,  1]
    ])

    # Test point in LiDAR coordinates
    lidar_point = np.array([10, 5, 1, 1])  # 10m forward, 5m left, 1m up (homogeneous)
    camera_point = lidar2camera @ lidar_point

    print(f"LiDAR point: {lidar_point[:3]}")
    print(f"Camera point: {camera_point[:3]}")

    if camera_point[2] > 0:
        u = fx * camera_point[0] / camera_point[2] + cx
        v = fy * camera_point[1] / camera_point[2] + cy
        print(f"Projected to: ({u:.1f}, {v:.1f})")
    else:
        print(f"Behind camera (z={camera_point[2]:.1f})")

if __name__ == "__main__":
    test_projection()