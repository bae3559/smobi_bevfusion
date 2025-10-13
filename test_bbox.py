#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add bevfusion to path
sys.path.insert(0, '/home/sum/bevfusion')

try:
    # Try to import without the full environment setup
    import torch
    print("PyTorch imported successfully")

    # Create a simple test bbox
    # Format: [x, y, z, dx, dy, dz, heading, ...]
    test_bbox = np.array([[10.0, 5.0, 1.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]])
    print(f"Test bbox: {test_bbox[0]}")
    print(f"Position: x={test_bbox[0,0]}, y={test_bbox[0,1]}, z={test_bbox[0,2]}")
    print(f"Dimensions: dx={test_bbox[0,3]}, dy={test_bbox[0,4]}, dz={test_bbox[0,5]}")
    print(f"Heading: {test_bbox[0,6]} radians")

    # Adjust Z coordinate (move bottom center to center)
    test_bbox[..., 2] -= test_bbox[..., 5] / 2  # z -= dz/2
    print(f"Adjusted bbox (bottom center): {test_bbox[0]}")

    # Try to create LiDARInstance3DBoxes
    try:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        bbox_tensor = torch.from_numpy(test_bbox).float()
        lidar_boxes = LiDARInstance3DBoxes(bbox_tensor, box_dim=9)

        print(f"LiDARInstance3DBoxes created successfully")
        print(f"Corners shape: {lidar_boxes.corners.shape}")

        corners = lidar_boxes.corners[0].numpy()  # First box corners
        print(f"8 corners of the test box:")
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: x={corner[0]:.2f}, y={corner[1]:.2f}, z={corner[2]:.2f}")

        # Check dimensions
        x_range = corners[:, 0].max() - corners[:, 0].min()
        y_range = corners[:, 1].max() - corners[:, 1].min()
        z_range = corners[:, 2].max() - corners[:, 2].min()
        print(f"Corner ranges: x={x_range:.2f}, y={y_range:.2f}, z={z_range:.2f}")

    except ImportError as e:
        print(f"Could not import LiDARInstance3DBoxes: {e}")

except ImportError as e:
    print(f"Could not import PyTorch: {e}")