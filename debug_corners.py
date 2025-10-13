#!/usr/bin/env python3

import numpy as np

def calculate_bbox_corners(center, dims, yaw):
    """
    Calculate 8 corners of a 3D bounding box
    Args:
        center: [x, y, z] center of the box
        dims: [dx, dy, dz] dimensions of the box
        yaw: rotation around z-axis
    Returns:
        corners: 8x3 array of corner coordinates
    """
    x, y, z = center
    dx, dy, dz = dims

    # Create 8 corners in local coordinate system (relative to center)
    # Using the same logic as LiDARInstance3DBoxes
    corners_norm = np.array([
        [0, 0, 0],  # 0: x0, y0, z0
        [0, 0, 1],  # 1: x0, y0, z1
        [0, 1, 0],  # 2: x0, y1, z0
        [0, 1, 1],  # 3: x0, y1, z1
        [1, 0, 0],  # 4: x1, y0, z0
        [1, 0, 1],  # 5: x1, y0, z1
        [1, 1, 0],  # 6: x1, y1, z0
        [1, 1, 1]   # 7: x1, y1, z1
    ])

    # Reorder according to LiDARInstance3DBoxes: [0, 1, 3, 2, 4, 5, 7, 6]
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]

    # Convert to box dimensions (relative to center)
    corners_norm = corners_norm - 0.5  # Use relative origin [0.5, 0.5, 0.5]
    corners = corners_norm * dims

    # Apply rotation around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    corners = corners @ rotation_matrix.T

    # Translate to world position
    corners += center

    return corners

# Test with a simple box
center = [10.0, 5.0, 1.0]  # 10m forward, 5m left, 1m up
dims = [4.0, 2.0, 1.5]     # 4m long, 2m wide, 1.5m tall
yaw = 0.0                   # No rotation

print("Test bbox:")
print(f"Center: {center}")
print(f"Dimensions: {dims}")
print(f"Yaw: {yaw}")

corners = calculate_bbox_corners(center, dims, yaw)

print(f"\n8 corners:")
for i, corner in enumerate(corners):
    print(f"  Corner {i}: x={corner[0]:.2f}, y={corner[1]:.2f}, z={corner[2]:.2f}")

# Check dimensions
x_range = corners[:, 0].max() - corners[:, 0].min()
y_range = corners[:, 1].max() - corners[:, 1].min()
z_range = corners[:, 2].max() - corners[:, 2].min()
print(f"\nCorner ranges: x={x_range:.2f}, y={y_range:.2f}, z={z_range:.2f}")
print(f"Expected ranges: x={dims[0]:.2f}, y={dims[1]:.2f}, z={dims[2]:.2f}")

# Test corner connections for rectangular shape
print(f"\nBottom face (z0) corners:")
bottom_indices = [0, 2, 6, 4]  # Should form a rectangle in xy plane
for i in bottom_indices:
    print(f"  Corner {i}: x={corners[i, 0]:.2f}, y={corners[i, 1]:.2f}, z={corners[i, 2]:.2f}")

print(f"\nTop face (z1) corners:")
top_indices = [1, 3, 7, 5]  # Should form a rectangle in xy plane
for i in top_indices:
    print(f"  Corner {i}: x={corners[i, 0]:.2f}, y={corners[i, 1]:.2f}, z={corners[i, 2]:.2f}")