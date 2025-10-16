import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map", "create_collage"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    # Waymo classes
    "vehicle": (255, 158, 0),    # Same as car
    "cyclist": (220, 20, 60),    # Same as bicycle
    "sign": (47, 79, 79),        # Same as traffic_cone
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
    dataset_type,
    lidar_points: Optional[np.ndarray] = None,
    ) -> None:
    '''
    # Expand canvas size to show boxes that go outside image bounds
    original_h, original_w = image.shape[:2]
    expanded_w = original_w * 2  # Double width
    expanded_h = original_h * 2  # Double height

    # Create expanded black canvas
    canvas = np.zeros((expanded_h, expanded_w, 3), dtype=np.uint8)

    # Place original image in the center
    start_y = expanded_h // 4
    start_x = expanded_w // 4
    canvas[start_y:start_y+original_h, start_x:start_x+original_w] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f"Expanded canvas size: {expanded_w}x{expanded_h} (original: {original_w}x{original_h})")
    print(f"Original image placed at offset: ({start_x}, {start_y})")
    '''
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    if dataset_type=="waymo":
        if bboxes is not None and len(bboxes) > 0:
            corners = bboxes.corners
            num_bboxes = corners.shape[0]

            coords = np.concatenate(
                [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
            )
            transform = copy.deepcopy(transform).reshape(4, 4)

            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)

            indices = np.all(coords[..., 2] > 0, axis=1)
            # print(f"  Before depth filter: {coords.shape[0]} boxes")
            # print(f"  After depth filter: {np.sum(indices)} boxes")
            coords = coords[indices]
            labels = labels[indices]

            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            labels = labels[indices]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]

            coords = coords[..., :2].reshape(-1, 8, 2)
            # print(f"  Final coords shape: {coords.shape}")
            # print(f"  Coords range: x=[{coords[..., 0].min():.1f}, {coords[..., 0].max():.1f}], y=[{coords[..., 1].min():.1f}, {coords[..., 1].max():.1f}]")
            # print(f"  Image shape: {canvas.shape}")
            for index in range(coords.shape[0]):
                name = classes[labels[index]]
                # print(f"  Drawing box {index} for class '{name}'")
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        canvas,
                        tuple(coords[index, start].astype(np.int32)),
                        tuple(coords[index, end].astype(np.int32)),
                        color or OBJECT_PALETTE[name],
                        thickness,
                        cv2.LINE_AA,
                    )
            canvas = canvas.astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath)
    else:
        if bboxes is not None and len(bboxes) > 0:
            corners = bboxes.corners
            num_bboxes = corners.shape[0]

            coords = np.concatenate(
                [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
            )
            transform = copy.deepcopy(transform).reshape(4, 4)

            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)


            indices = np.all(coords[..., 2] > 0, axis=1)
            # print(f"  Before depth filter: {coords.shape[0]} boxes")
            # print(f"  After depth filter: {np.sum(indices)} boxes")
            coords = coords[indices]
            labels = labels[indices]

            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            labels = labels[indices]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]

            coords = coords[..., :2].reshape(-1, 8, 2)
            # print(f"  Final coords shape: {coords.shape}")
            # print(f"  Coords range: x=[{coords[..., 0].min():.1f}, {coords[..., 0].max():.1f}], y=[{coords[..., 1].min():.1f}, {coords[..., 1].max():.1f}]")
            # print(f"  Image shape: {canvas.shape}")
            for index in range(coords.shape[0]):
                name = classes[labels[index]]
                # print(f"  Drawing box {index} for class '{name}'")
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        canvas,
                        tuple(coords[index, start].astype(np.int32)),
                        tuple(coords[index, end].astype(np.int32)),
                        color or OBJECT_PALETTE[name],
                        thickness,
                        cv2.LINE_AA,
                    )
            canvas = canvas.astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath)



def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]


        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def create_collage(
    fpath: str,
    camera_images: List[str],  # List of camera image paths
    lidar_image: str,  # LiDAR image path
    layout: str = "3x3"  # Layout configuration
) -> None:
    """Create a collage of camera and LiDAR images.

    Args:
        fpath: Output path for the collage
        camera_images: List of paths to camera images (should be 6 for Waymo)
        lidar_image: Path to LiDAR visualization image
        layout: Layout configuration ("3x3", "2x4", etc.)
    """
    import cv2
    import numpy as np

    # Read all images
    images = []

    # Read camera images
    for img_path in camera_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                # Create placeholder if image doesn't exist
                images.append(np.zeros((640, 960, 3), dtype=np.uint8))
        else:
            # Create placeholder if file doesn't exist
            images.append(np.zeros((640, 960, 3), dtype=np.uint8))

    # Read LiDAR image
    if os.path.exists(lidar_image):
        lidar_img = cv2.imread(lidar_image)
        if lidar_img is not None:
            images.append(lidar_img)
        else:
            images.append(np.zeros((640, 960, 3), dtype=np.uint8))
    else:
        images.append(np.zeros((640, 960, 3), dtype=np.uint8))

    # Resize all images to consistent size
    target_size = (480, 320)  # width, height
    resized_images = []
    for img in images:
        resized = cv2.resize(img, target_size)
        resized_images.append(resized)

    # Create collage based on layout
    if layout == "3x3":
        # 3x3 grid layout
        rows = []
        for i in range(3):
            if i < 2:
                # First two rows: 3 camera images each
                row_images = resized_images[i*3:(i+1)*3]
                while len(row_images) < 3:
                    row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            else:
                # Third row: 1 LiDAR image in center
                if len(resized_images) > 6:
                    row_images = [
                        np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8),
                        resized_images[6],  # LiDAR image
                        np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    ]
                else:
                    row_images = [np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)] * 3

            row = np.hstack(row_images)
            rows.append(row)

        collage = np.vstack(rows)

    elif layout == "2x4":
        # 2x4 grid layout (2 rows, 4 columns)
        rows = []
        for i in range(2):
            if i == 0:
                # First row: 4 camera images
                row_images = resized_images[i*4:(i+1)*4]
                while len(row_images) < 4:
                    row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            else:
                # Second row: 2 camera images + LiDAR + empty
                row_images = resized_images[4:6]  # Last 2 camera images
                if len(resized_images) > 6:
                    row_images.append(resized_images[6])  # LiDAR
                else:
                    row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))  # Empty slot

            row = np.hstack(row_images)
            rows.append(row)

        collage = np.vstack(rows)

    elif layout == "lidar_left":
        # Left side: Large LiDAR image, Right side: 2x3 grid of cameras
        # LiDAR target area size
        lidar_area_width = target_size[0] * 2  # 960
        lidar_area_height = target_size[1] * 2  # 640

        if len(resized_images) > 6:
            # Keep original LiDAR image aspect ratio
            original_lidar = images[6]
            orig_h, orig_w = original_lidar.shape[:2]
            orig_ratio = orig_w / orig_h

            # Calculate new size while maintaining aspect ratio
            if orig_ratio > (lidar_area_width / lidar_area_height):
                # Image is wider - fit to width
                new_width = lidar_area_width
                new_height = int(lidar_area_width / orig_ratio)
            else:
                # Image is taller - fit to height
                new_height = lidar_area_height
                new_width = int(lidar_area_height * orig_ratio)

            # Resize maintaining aspect ratio
            lidar_resized = cv2.resize(original_lidar, (new_width, new_height))

            # Create a centered image in the target area
            lidar_canvas = np.zeros((lidar_area_height, lidar_area_width, 3), dtype=np.uint8)

            # Center the resized image
            start_y = (lidar_area_height - new_height) // 2
            start_x = (lidar_area_width - new_width) // 2
            lidar_canvas[start_y:start_y+new_height, start_x:start_x+new_width] = lidar_resized

            lidar_resized = lidar_canvas
        else:
            lidar_resized = np.zeros((lidar_area_height, lidar_area_width, 3), dtype=np.uint8)

        # Camera grid (2x3) on the right side
        camera_rows = []
        for i in range(2):  # 2 rows
            row_images = []
            for j in range(3):  # 3 columns
                idx = i * 3 + j
                if idx < len(resized_images) and idx < 6:  # Only camera images
                    row_images.append(resized_images[idx])
                else:
                    row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

            camera_row = np.hstack(row_images)
            camera_rows.append(camera_row)

        camera_grid = np.vstack(camera_rows)

        # Combine LiDAR (left) and camera grid (right)
        collage = np.hstack([lidar_resized, camera_grid])

    # Add labels to each image section
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    # Camera labels (Waymo order)
    camera_labels = ["FRONT", "FRONT_LEFT", "SIDE_LEFT",
                     "FRONT_RIGHT", "SIDE_RIGHT", "BACK"]

    if layout == "3x3":
        # Add labels for 3x3 layout
        for i in range(2):  # First two rows
            for j in range(3):
                idx = i * 3 + j
                if idx < len(camera_labels):
                    x = j * target_size[0] + 10
                    y = i * target_size[1] + 30
                    cv2.putText(collage, camera_labels[idx], (x, y), font, font_scale, color, thickness)

        # LiDAR label (center of third row)
        x = target_size[0] + 10
        y = 2 * target_size[1] + 30
        cv2.putText(collage, "LiDAR", (x, y), font, font_scale, color, thickness)

    elif layout == "2x4":
        # Add labels for 2x4 layout
        for i in range(4):  # First row
            if i < len(camera_labels):
                x = i * target_size[0] + 10
                y = 30
                cv2.putText(collage, camera_labels[i], (x, y), font, font_scale, color, thickness)

        # Second row labels
        for i in range(2):  # Two more cameras
            idx = 4 + i
            if idx < len(camera_labels):
                x = i * target_size[0] + 10
                y = target_size[1] + 30
                cv2.putText(collage, camera_labels[idx], (x, y), font, font_scale, color, thickness)

        # LiDAR label
        x = 2 * target_size[0] + 10
        y = target_size[1] + 30
        cv2.putText(collage, "LiDAR", (x, y), font, font_scale, color, thickness)

    elif layout == "lidar_left":
        # LiDAR label (left side, top-left corner)
        cv2.putText(collage, "LiDAR", (10, 30), font, font_scale, color, thickness)

        # Camera labels (right side, 2x3 grid)
        lidar_width = target_size[0] * 2  # Width of LiDAR section
        for i in range(2):  # 2 rows
            for j in range(3):  # 3 columns
                idx = i * 3 + j
                if idx < len(camera_labels):
                    x = lidar_width + j * target_size[0] + 10
                    y = i * target_size[1] + 30
                    cv2.putText(collage, camera_labels[idx], (x, y), font, font_scale, color, thickness)

    # Save collage
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    cv2.imwrite(fpath, collage)
    print(f"Collage saved to: {fpath}")
    print(f"Collage size: {collage.shape[1]}x{collage.shape[0]}")