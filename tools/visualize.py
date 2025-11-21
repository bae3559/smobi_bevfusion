import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
#from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map, create_collage
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    # Force test_mode=True and use test pipeline for visualization (no augmentation)
    if args.split == "train":
        # IMPORTANT: Use train split but with val pipeline (no augmentation)
        # This ensures we visualize original images without rotation/flip/crop
        dataset_cfg = cfg.data.train.copy()

        # If wrapped in CBGSDataset, unwrap it and modify the inner dataset
        if dataset_cfg.get('type') == 'CBGSDataset':
            print("[INFO] Detected CBGSDataset wrapper, unwrapping...")
            dataset_cfg = dataset_cfg['dataset'].copy()

        dataset_cfg['test_mode'] = False  # Keep test_mode=False to load GT annotations
        dataset_cfg['pipeline'] = cfg.data.val.pipeline  # Use val pipeline (no augmentation)
        print("[INFO] Using train split with validation pipeline (no augmentation)")
        print(f"[DEBUG] Dataset type: {dataset_cfg.get('type')}")
        print(f"[DEBUG] Pipeline steps:")
        for i, step in enumerate(dataset_cfg['pipeline']):
            print(f"  {i}: {step['type']} - is_train={step.get('is_train', 'N/A')}")
    else:
        dataset_cfg = cfg.data[args.split]

    dataset = build_dataset(dataset_cfg)
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for i, data in enumerate(tqdm(dataflow)):

        metas = data["metas"].data[0][0]

        # Get camera_intrinsics from data (it's in meta_keys, not meta_lis_keys)
        # DataContainer wraps the data: .data[0] gets batch, [0] gets first sample
        camera_intrinsics = data["camera_intrinsics"].data[0][0] if "camera_intrinsics" in data else None
        # Handle both nuScenes (has token) and Waymo (doesn't have token)
        if "token" in metas:
            name = "{}-{}".format(metas["timestamp"], metas["token"])
            dataset_type = "nuscenes"
        else:
            # For Waymo, extract frame identifier from lidar_path or use timestamp
            lidar_path = metas.get("lidar_path", "")
            if lidar_path:
                # Extract frame ID from path like: .../1522688014970187.bin
                frame_id = os.path.basename(lidar_path).split('.')[0]
                name = "{}-{}".format(metas["timestamp"], frame_id)
            else:
                # Fallback to just timestamp
                name = str(metas["timestamp"])
            dataset_type = "waymo"

        if args.mode == "pred":
            try:
                with torch.inference_mode():
                    # Try to get model predictions
                    data_list = list(data.keys())
                    print(f"[DEBUG] {len(data_list)} Data Keys: {data_list}")
                    # Temporarily set test_mode to avoid some errors
                    model.module.test_cfg = getattr(model.module, 'test_cfg', {})
                    outputs = model(**data, return_loss=True)
                print("Prediction successful!")
            except Exception as e:
                print(f"Prediction failed: {e}")
                print("Falling back to GT visualization")
                args.mode = "gt"  # Switch to GT mode
                outputs = None

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            # Debug: check bbox before processing
            raw_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            print(f"[DEBUG] Dataset type: {dataset_type}")
            print(f"[DEBUG] Raw bbox shape: {raw_bboxes.shape}")
            if len(raw_bboxes) > 0:
                print(f"[DEBUG] First raw bbox: {raw_bboxes[0]}")
                print(f"[DEBUG] Bbox X range: [{raw_bboxes[:, 0].min():.2f}, {raw_bboxes[:, 0].max():.2f}]")
                print(f"[DEBUG] Bbox Y range: [{raw_bboxes[:, 1].min():.2f}, {raw_bboxes[:, 1].max():.2f}]")
                print(f"[DEBUG] Bbox Z range: [{raw_bboxes[:, 2].min():.2f}, {raw_bboxes[:, 2].max():.2f}]")

            if dataset_type=="waymo":
                bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()

                # dx_mean = float(bboxes[:, 3].mean())
                # dy_mean = float(bboxes[:, 4].mean())
                # print("mean dx:", dx_mean, " mean dy:", dy_mean)  # 보통 차량은 dx > dy

                #print("boxes before LiDARInstance3DBoxes",bboxes )
                labels = data["gt_labels_3d"].data[0][0].numpy()

                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    labels = labels[indices]

                bboxes[..., 2] -= bboxes[..., 5] / 2

                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

                # Debug: print corners after LiDARInstance3DBoxes conversion
                if len(bboxes) > 0:
                    #print(f"[DEBUG] After LiDARInstance3DBoxes conversion:")
                    #print(f"[DEBUG] First box corners shape: {bboxes.corners.shape}")
                    corners_0 = bboxes.corners[0]
                    #print(f"[DEBUG] First box corner ranges:")
                    #print(f"  X: [{corners_0[:, 0].min():.2f}, {corners_0[:, 0].max():.2f}]")
                    #print(f"  Y: [{corners_0[:, 1].min():.2f}, {corners_0[:, 1].max():.2f}]")
                    #print(f"  Z: [{corners_0[:, 2].min():.2f}, {corners_0[:, 2].max():.2f}]")

                print(f"Processing {len(bboxes)} boxes for visualization")
            else:
                # For nuScenes/mantruck: data already contains LiDARInstance3DBoxes
                # Don't need to re-process, just use it directly
                bboxes = data["gt_bboxes_3d"].data[0][0]
                labels = data["gt_labels_3d"].data[0][0].numpy()
                if args.bbox_classes is not None:
                    indices = np.isin(labels, args.bbox_classes)
                    bboxes = bboxes[indices]
                    labels = labels[indices]

                # bboxes is already LiDARInstance3DBoxes, no need to convert

                # Debug: print corners after LiDARInstance3DBoxes conversion
                if len(bboxes) > 0:
                    print(f"[DEBUG] After LiDARInstance3DBoxes conversion:")
                    print(f"[DEBUG] First box corners shape: {bboxes.corners.shape}")
                    corners_0 = bboxes.corners[0]
                    print(f"[DEBUG] First box corner ranges:")
                    print(f"  X: [{corners_0[:, 0].min():.2f}, {corners_0[:, 0].max():.2f}]")
                    print(f"  Y: [{corners_0[:, 1].min():.2f}, {corners_0[:, 1].max():.2f}]")
                    print(f"  Z: [{corners_0[:, 2].min():.2f}, {corners_0[:, 2].max():.2f}]")

                print(f"Processing {len(bboxes)} boxes for visualization")
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        # Store image paths for collage
        camera_image_paths = []
        lidar_image_path = None
        #print("metas",metas)
        # meta {'filename': ['data/waymo/Waymo_processed/images/CAM_FRONT/10203656353524179475_7625_000_7645_000_1522688015269987.jpg', 'data/waymo/Waymo_processed/images/CAM_FRONT_LEFT/10203656353524179475_7625_000_7645_000_1522688015269987.jpg', 'data/waymo/Waymo_processed/images/CAM_BACK_LEFT/10203656353524179475_7625_000_7645_000_1522688015269987.jpg', 'data/waymo/Waymo_processed/images/CAM_FRONT_RIGHT/10203656353524179475_7625_000_7645_000_1522688015269987.jpg', 
        # 'data/waymo/Waymo_processed/images/CAM_BACK_RIGHT/10203656353524179475_7625_000_7645_000_1522688015269987.jpg', 
        # 'data/waymo/Waymo_processed/images/CAM_BACK/10203656353524179475_7625_000_7645_000_1522688015269987.jpg'], 
        # 'timestamp': 1522688015269987, 'ori_shape': (1920, 1280), 'img_shape': (1920, 1280),
        #  'lidar2image': [array([[-1.8736277e+03,  1.2657885e+03,  3.7847519e+01, -1.5178189e+02],
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                camera_output_path = os.path.join(args.out_dir, f"camera-{k}", f"{name}.png")
                camera_image_paths.append(camera_output_path)

                visualize_camera(
                    camera_output_path,
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    camera_distortions=metas["camera_distortions"][k] if "camera_distortions" in metas else None,
                    camera_intrinsics=camera_intrinsics[k].numpy() if camera_intrinsics is not None else None,
                    classes=cfg.object_classes,
                    dataset_type=dataset_type
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()

            # Debug: print LiDAR point cloud statistics
            # print(f"[DEBUG] LiDAR points shape: {lidar.shape}")
            # print(f"[DEBUG] X range: [{lidar[:, 0].min():.2f}, {lidar[:, 0].max():.2f}]")
            # print(f"[DEBUG] Y range: [{lidar[:, 1].min():.2f}, {lidar[:, 1].max():.2f}]")
            # print(f"[DEBUG] Z range: [{lidar[:, 2].min():.2f}, {lidar[:, 2].max():.2f}]")
            # print(f"[DEBUG] Config point_cloud_range: {cfg.point_cloud_range}")

            lidar_image_path = os.path.join(args.out_dir, "lidar", f"{name}.png")

            # Check if we need to swap X/Y for mantruck dataset
            swap_xy = (dataset_type == "nuscenes" and
                      hasattr(cfg, 'dataset_type') and
                      'mantruck' in cfg.dataset_type.lower())

            visualize_lidar(
                lidar_image_path,
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
                swap_xy=swap_xy,
            )

        # Create collage if we have both camera and lidar images
        if camera_image_paths and lidar_image_path:
            collage_path = os.path.join(args.out_dir, "collage", f"{name}.png")
            create_collage(
                collage_path,
                camera_image_paths,
                lidar_image_path,
                layout="lidar_left"  # Left: large LiDAR, Right: 2x3 cameras
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
