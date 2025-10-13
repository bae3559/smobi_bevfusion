#!/usr/bin/env python3

import numpy as np

def test_waymo_matrices():
    """Test typical Waymo transformation matrices"""

    print("=== Waymo Coordinate System Analysis ===")

    # Typical Waymo camera intrinsics
    fx, fy = 2050.0, 2050.0
    cx, cy = 960.0, 640.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    print("Camera intrinsics K:")
    print(K)

    # Make 4x4 version
    cam_intrinsic = np.eye(4, dtype=np.float32)
    cam_intrinsic[:3, :3] = K
    print("\nCamera intrinsics (4x4):")
    print(cam_intrinsic)

    # Simulate a typical front camera extrinsic (camera2ego)
    # Waymo front camera is typically mounted on top, looking forward
    # Let's say 2m high, looking slightly down
    camera2ego = np.array([
        [1.0,  0.0,  0.0,  0.0],   # x: forward (same as ego)
        [0.0,  1.0,  0.0,  0.0],   # y: left (same as ego)
        [0.0,  0.0,  1.0,  2.0],   # z: up (camera is 2m high)
        [0.0,  0.0,  0.0,  1.0]
    ])
    print("\nCamera2Ego (example):")
    print(camera2ego)

    # Simulate lidar2ego (identity for simplicity)
    lidar2ego = np.eye(4)
    print("\nLiDAR2Ego (identity):")
    print(lidar2ego)

    # Calculate camera2lidar = inv(lidar2ego) @ camera2ego
    camera2lidar = np.linalg.inv(lidar2ego) @ camera2ego
    print("\nCamera2LiDAR:")
    print(camera2lidar)

    # Calculate lidar2camera = inv(camera2lidar)
    lidar2camera = np.linalg.inv(camera2lidar)
    print("\nLiDAR2Camera:")
    print(lidar2camera)

    # Calculate lidar2image = K @ lidar2camera
    lidar2image = cam_intrinsic @ lidar2camera
    print("\nLiDAR2Image:")
    print(lidar2image)

    # Test projection with a point in front of the vehicle
    print("\n=== Test Projection ===")

    # Point 10m ahead, 2m left, 1m up in LiDAR coordinates
    lidar_point = np.array([10.0, 2.0, 1.0, 1.0])  # homogeneous
    print(f"LiDAR point: {lidar_point[:3]}")

    # Transform to camera coordinates
    camera_point = lidar2camera @ lidar_point
    print(f"Camera point: {camera_point[:3]}")

    # Project to image
    if camera_point[2] > 0:
        image_point = lidar2image @ lidar_point
        u = image_point[0] / image_point[2]
        v = image_point[1] / image_point[2]
        print(f"Image point: ({u:.1f}, {v:.1f})")

        # Check bounds
        in_bounds = 0 <= u < 1920 and 0 <= v < 1280
        print(f"In bounds: {in_bounds}")
    else:
        print(f"Behind camera (z={camera_point[2]:.1f})")

    # Test with a point very far away
    print(f"\n=== Test Far Point ===")
    far_point = np.array([100.0, 50.0, 5.0, 1.0])  # 100m ahead, 50m left, 5m up
    print(f"Far LiDAR point: {far_point[:3]}")

    camera_far = lidar2camera @ far_point
    print(f"Far camera point: {camera_far[:3]}")

    if camera_far[2] > 0:
        image_far = lidar2image @ far_point
        u_far = image_far[0] / image_far[2]
        v_far = image_far[1] / image_far[2]
        print(f"Far image point: ({u_far:.1f}, {v_far:.1f})")
        print(f"Way outside bounds (expected for such a far point)")

if __name__ == "__main__":
    test_waymo_matrices()