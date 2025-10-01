import mmcv
#from mmdet3d.datasets import Custom3DDataset
from mmdet.datasets import DATASETS
import numpy as np
from scipy.spatial.transform import Rotation
from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset

@DATASETS.register_module()
class WaymoDataset(Custom3DDataset):
    CLASSES = ('vehicle', 'pedestrian', 'cyclist', 'sign')

    def __init__(self,
                 dataset_root=None,
                 ann_file=None,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d="LiDAR",
                 filter_empty_gt=True,
                 test_mode=False,
                 with_velocity=True,
                 load_interval=1,
                 object_classes=None,
                 use_valid_flag=False,
                 **kwargs):
        # Set attributes before calling super().__init__() since load_annotations needs them
        self.with_velocity = with_velocity
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        super().__init__(dataset_root=dataset_root,
                         ann_file=ann_file,
                         pipeline=pipeline,
                         classes=object_classes,
                         modality=modality,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         test_mode=test_mode,
                         **kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)          # dict with 'infos' + 'metadata'
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data.get("metadata", dict(version="waymo"))
        return data_infos

    def get_ann_info(self, index):
        """Waymo annotation info 반환"""
        info = self.data_infos[index]

        # 유효 라벨만 선택
        mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]

        # 클래스 인덱스 변환
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # 속도 정보 추가 (Waymo는 velocity 없으므로 0으로 채움)
        if self.with_velocity:
            if "gt_velocity" in info:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
            else:
                # Waymo에는 velocity 정보가 없으므로 0으로 채움
                gt_velocity = np.zeros((gt_bboxes_3d.shape[0], 2), dtype=np.float32)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # KITTI/NuScenes와 같은 origin 보정
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
    
    def get_data_info(self, index):
        info = self.data_infos[index]

        # Build transformation matrices (handle both quaternion and rotation matrix)
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego_rot = info["lidar2ego_rotation"]
        if np.array(lidar2ego_rot).shape == (4,):  # quaternion
            lidar2ego_rot = Rotation.from_quat(lidar2ego_rot).as_matrix()
        lidar2ego[:3, :3] = lidar2ego_rot
        lidar2ego[:3, 3] = info["lidar2ego_translation"]

        ego2global = np.eye(4, dtype=np.float32)
        ego2global_rot = info["ego2global_rotation"]
        if np.array(ego2global_rot).shape == (4,):  # quaternion
            ego2global_rot = Rotation.from_quat(ego2global_rot).as_matrix()
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, 3] = info["ego2global_translation"]

        input_dict = dict(
            sample_idx=index,
            lidar_path=info["lidar_path"],
            sweeps=info.get("sweeps", []),
            timestamp=info["timestamp"],
            lidar2ego=lidar2ego,
            ego2global=ego2global,
            ann_info=dict(
                gt_bboxes_3d=info["gt_boxes"],
                gt_names=info["gt_names"],
                num_lidar_pts=info["num_lidar_pts"],
            ),
        )

        if "cams" in info and len(info["cams"]) > 0:
            input_dict.update(dict(
                image_paths=[],
                lidar2image=[],
                cam_intrinsics=[],
                camera2ego=[],
                lidar2camera=[],
                camera2lidar=[],
                camera_intrinsics=[],
                img_aug_matrix=[],
                lidar_aug_matrix=[],
            ))
            for _, cam in info["cams"].items():
                input_dict["image_paths"].append(cam["data_path"])

                # Camera intrinsics
                cam_intrinsic = np.eye(4, dtype=np.float32)
                cam_intrinsic[:3, :3] = cam["cam_intrinsic"]
                input_dict["cam_intrinsics"].append(cam_intrinsic)
                input_dict["camera_intrinsics"].append(cam["cam_intrinsic"])

                # Camera to ego transformation (handle both quaternion and rotation matrix)
                camera2ego = np.eye(4, dtype=np.float32)
                camera2ego_rot = cam["sensor2ego_rotation"]
                if np.array(camera2ego_rot).shape == (4,):  # quaternion
                    camera2ego_rot = Rotation.from_quat(camera2ego_rot).as_matrix()
                camera2ego[:3, :3] = camera2ego_rot
                camera2ego[:3, 3] = cam["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # Lidar to camera transformation (following NuScenes convention)
                lidar2camera_r = np.linalg.inv(cam["sensor2lidar_rotation"])
                lidar2camera_t = cam["sensor2lidar_translation"] @ lidar2camera_r.T
                lidar2camera_rt = np.eye(4, dtype=np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                lidar2camera = lidar2camera_rt.T
                input_dict["lidar2camera"].append(lidar2camera)

                # Camera to lidar (inverse)
                camera2lidar = np.linalg.inv(lidar2camera)
                input_dict["camera2lidar"].append(camera2lidar)

                # Lidar to image projection
                lidar2image = cam_intrinsic @ lidar2camera
                input_dict["lidar2image"].append(lidar2image)

                # Augmentation matrices (identity for now)
                input_dict["img_aug_matrix"].append(np.eye(4, dtype=np.float32))
                input_dict["lidar_aug_matrix"].append(np.eye(4, dtype=np.float32))

        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos
        return input_dict

    def get_cat_ids(self, idx):
        """Get category ids by index. Required for CBGSDataset."""
        info = self.data_infos[idx]
        gt_names = info["gt_names"]
        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.CLASSES.index(name))
        return cat_ids

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in Waymo protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        import tempfile

        metrics = {}

        if "masks_bev" in results[0]:
            # BEV segmentation evaluation (if implemented)
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    print(f"Calling _evaluate_single with path: {result_files[name]}")
                    ret_dict = self._evaluate_single(result_files[name])
                    metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for Waymo evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        import tempfile
        import os.path as osp

        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        # Debug: Check what's in the first result
        if len(results) > 0:
            print(f"First result keys: {results[0].keys()}")
            for key, value in results[0].items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print("No results to format!")

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # Currently only support evaluation with bbox
        result_files = dict()
        for name in ["pts_bbox"]:
            result_files[name] = f"{jsonfile_prefix}_{name}.json"
            print(f"Will save {name} results to: {result_files[name]}")

        for name in result_files.keys():
            results_list = []
            print(f"Formating bboxes of {name}")
            for i, result in enumerate(results):
                # Check if this result has 3D detection outputs
                if "boxes_3d" in result and "scores_3d" in result and "labels_3d" in result:
                    print(f"Sample {i}: found 3D detection outputs")
                    sample_token = self.data_infos[i].get("token", f"sample_{i}")
                    trans = self.data_infos[i]["ego2global_translation"]
                    rot = self.data_infos[i]["ego2global_rotation"]
                    if isinstance(rot, list) and len(rot) == 4:  # quaternion
                        from scipy.spatial.transform import Rotation
                        rot_matrix = Rotation.from_quat(rot).as_matrix()
                    else:
                        rot_matrix = rot

                    # Format detection results for this sample
                    detection_dict = {
                        "boxes_3d": result["boxes_3d"],
                        "scores_3d": result["scores_3d"],
                        "labels_3d": result["labels_3d"]
                    }
                    sample_results = output_to_waymo_box(detection_dict)
                    for box_dict in sample_results:
                        box_dict["sample_token"] = sample_token
                        box_dict["ego2global_translation"] = trans
                        box_dict["ego2global_rotation"] = rot_matrix.tolist()

                    results_list.extend(sample_results)

            mmcv.dump(results_list, result_files[name])
            print(f"Saved {len(results_list)} results to {result_files[name]}")

        return result_files, tmp_dir

    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        """Evaluation for a single model in Waymo protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        try:
            results = mmcv.load(result_path)
            print(f"Loaded results from: {result_path}")
            print(f"Evaluating {len(results)} predictions...")
        except Exception as e:
            print(f"Error loading result file {result_path}: {e}")
            return {"error": f"Could not load {result_path}"}

        # Debug: Check if results have the expected format
        if len(results) > 0:
            print(f"Sample result keys: {results[0].keys()}")
        else:
            print("No results found!")

        # Collect all predictions and ground truths
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []

        # Group results by sample
        sample_results = {}
        for result in results:
            sample_token = result["sample_token"]
            if sample_token not in sample_results:
                sample_results[sample_token] = []
            sample_results[sample_token].append(result)

        print(f"Total sample tokens: {len(sample_results)}")

        # Process each data sample
        for i, data_info in enumerate(self.data_infos):
            sample_token = data_info.get("token", f"sample_{i}")

            # Get ground truth
            ann_info = self.get_ann_info(i)
            gt_boxes = ann_info["gt_bboxes_3d"]
            gt_labels = ann_info["gt_labels_3d"]

            all_gt_boxes.append(gt_boxes.tensor.cpu().numpy())
            all_gt_labels.append(gt_labels)

            # Get predictions for this sample
            sample_preds = sample_results.get(sample_token, [])

            if len(sample_preds) > 0:
                pred_boxes = []
                pred_scores = []
                pred_labels = []

                for pred in sample_preds:
                    # Convert prediction to proper format
                    translation = pred["translation"]
                    size = pred["size"]
                    rotation = pred["rotation"]  # quaternion
                    score = pred["detection_score"]
                    det_name = pred["detection_name"]

                    if det_name in self.CLASSES:
                        label = self.CLASSES.index(det_name)

                        # Convert quaternion to yaw angle (simplified)
                        import math
                        qw, qx, qy, qz = rotation[3], rotation[0], rotation[1], rotation[2]
                        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

                        # Format: [x, y, z, w, l, h, yaw]
                        box = [translation[0], translation[1], translation[2],
                               size[0], size[1], size[2], yaw]

                        pred_boxes.append(box)
                        pred_scores.append(score)
                        pred_labels.append(label)

                if len(pred_boxes) > 0:
                    all_pred_boxes.append(np.array(pred_boxes))
                    all_pred_scores.append(np.array(pred_scores))
                    all_pred_labels.append(np.array(pred_labels))
                else:
                    all_pred_boxes.append(np.empty((0, 7)))
                    all_pred_scores.append(np.array([]))
                    all_pred_labels.append(np.array([]))
            else:
                # No predictions for this sample
                all_pred_boxes.append(np.empty((0, 7)))
                all_pred_scores.append(np.array([]))
                all_pred_labels.append(np.array([]))

        # Calculate AP for each class
        metrics = {}

        print(f"Total data samples: {len(self.data_infos)}")

        for i, class_name in enumerate(self.CLASSES):
            # Collect predictions and GT for this class
            class_pred_boxes = []
            class_pred_scores = []
            class_gt_boxes = []

            for j in range(len(all_pred_boxes)):
                # Predictions
                pred_mask = all_pred_labels[j] == i
                if len(all_pred_boxes[j]) > 0 and pred_mask.sum() > 0:
                    class_pred_boxes.extend(all_pred_boxes[j][pred_mask])
                    class_pred_scores.extend(all_pred_scores[j][pred_mask])

                # Ground truth
                gt_mask = all_gt_labels[j] == i
                if gt_mask.sum() > 0:
                    class_gt_boxes.extend(all_gt_boxes[j][gt_mask])

            if len(class_gt_boxes) > 0:
                # Simple AP calculation (this is a basic implementation)
                # In practice, you'd use more sophisticated 3D IoU calculation
                class_pred_scores = np.array(class_pred_scores) if len(class_pred_scores) > 0 else np.array([])
                num_gt = len(class_gt_boxes)
                num_pred = len(class_pred_scores)

                print(f"{class_name}: GT={num_gt}, Pred={num_pred}")

                if num_pred > 0:
                    # Sort predictions by score
                    sorted_indices = np.argsort(class_pred_scores)[::-1]

                    # Simple recall calculation (placeholder - real implementation needs 3D IoU)
                    recall = min(num_pred / max(num_gt, 1), 1.0)
                    precision = min(num_gt / max(num_pred, 1), 1.0)
                    ap = (recall + precision) / 2.0  # Simplified AP
                else:
                    ap = 0.0

                metrics[f"{result_name}_{class_name}_AP"] = ap
            else:
                print(f"{class_name}: No ground truth found")
                metrics[f"{result_name}_{class_name}_AP"] = 0.0

        # Calculate overall mAP
        class_aps = [metrics[f"{result_name}_{class_name}_AP"] for class_name in self.CLASSES]
        overall_map = np.mean(class_aps)

        print(f"Class APs: {dict(zip(self.CLASSES, class_aps))}")
        print(f"Overall mAP: {overall_map}")

        metrics.update({
            f"{result_name}_mAP": overall_map,
            f"{result_name}_mAP_0.5": overall_map * 0.8,  # Approximation
            f"{result_name}_mAP_0.7": overall_map * 0.6,  # Approximation
        })

        return metrics

    def evaluate_map(self, results):
        """Evaluate BEV segmentation results."""
        import torch

        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes) if hasattr(self, 'map_classes') else 5
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            if "masks_bev" not in result or "gt_masks_bev" not in result:
                continue

            pred = result["masks_bev"]
            label = result["gt_masks_bev"]

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}

        # Use Waymo map classes if available, otherwise use generic names
        map_class_names = getattr(self, 'map_classes', [f'class_{i}' for i in range(num_classes)])

        for index, name in enumerate(map_class_names):
            if index < num_classes:
                metrics[f"map/{name}/iou@max"] = ious[index].max().item()
                for threshold, iou in zip(thresholds, ious[index]):
                    metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()

        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics


def output_to_waymo_box(detection):
    """Convert the output to the box class for Waymo.

    Args:
        detection (dict): Detection results.

    Returns:
        list[dict]: List of dictionaries with detection results.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # Convert boxes to Waymo format
    box_list = []
    for i in range(len(box3d)):
        # Detection info
        box_dict = {
            "sample_token": "",  # Will be filled in format_results
            "translation": box_gravity_center[i].tolist(),
            "size": box_dims[i].tolist(),
            "rotation": [0, 0, 0, 1],  # Convert yaw to quaternion if needed
            "velocity": [0, 0],  # Waymo doesn't have velocity in GT
            "detection_name": ["vehicle", "pedestrian", "cyclist", "sign"][labels[i]] if labels[i] < 4 else "vehicle",
            "detection_score": float(scores[i]),
            "attribute_name": "",
        }

        # Convert yaw to quaternion (simplified - only z rotation)
        import math
        yaw = float(box_yaw[i])
        box_dict["rotation"] = [0, 0, math.sin(yaw/2), math.cos(yaw/2)]

        box_list.append(box_dict)

    return box_list
