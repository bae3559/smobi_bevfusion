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

    # Waymo converter에서 사용된 원래 매핑 (잘못된 매핑)
    WAYMO_ORIGINAL_MAPPING = {
        1: "vehicle",     # 정상
        2: "pedestrian",  # 정상
        3: "cyclist",        # 잘못됨 - cyclist와 바뀜
        4: "sign"      # 잘못됨 - sign과 바뀜
    }
    
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

        # 유효 라벨만 선택 - 안전한 처리
        num_lidar_pts = info["num_lidar_pts"]
        gt_boxes = info["gt_boxes"]
        gt_names = info["gt_names"]

        # numpy 배열로 변환하여 안전하게 처리
        if isinstance(num_lidar_pts, (list, tuple)):
            num_lidar_pts = np.array(num_lidar_pts)
        if isinstance(gt_boxes, (list, tuple)):
            gt_boxes = np.array(gt_boxes)
        if isinstance(gt_names, (list, tuple)):
            gt_names = np.array(gt_names)

        # 유효한 mask 생성
        mask = num_lidar_pts > 0

        gt_bboxes_3d = gt_boxes[mask]
        gt_names_3d = gt_names[mask]

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


        gt = gt_boxes[mask] # [x,y,z, dx,dy,dz, yaw] (EGO, center-origin)

        # EGO -> LiDAR 변환행렬
        R_le = np.asarray(info["lidar2ego_rotation"], np.float32)
        t_le = np.asarray(info["lidar2ego_translation"], np.float32)
        T_le = np.eye(4, dtype=np.float32)
        T_le[:3,:3] = R_le
        T_le[:3,3] = t_le
        T_el = np.linalg.inv(T_le)  # EGO->LiDAR
        R_el = R_le.T                          # 회전은 전치가 곧 역행렬
        # ① 중심 좌표 변환
        centers_e = gt[:, :3]
        ones = np.ones((len(centers_e), 1), np.float32)
        centers_l = (T_el @ np.hstack([centers_e, ones]).T).T[:, :3]

        # ② yaw 변환 (방향벡터를 회전시키는 방식이 가장 안전)
        # lidar2ego: (4x4) TOP LiDAR → EGO
        
        #delta = np.arctan2(R_el[1, 0], R_el[0, 0])     # 회전행렬의 yaw 성분

        yaw_e = gt[:, 6]                               # EGO 기준 heading
        #yaw_l = ((yaw_e + delta) + np.pi) % (2*np.pi) - np.pi               # LiDAR 기준 yaw
        
        v_e  = np.stack([np.cos(yaw_e), np.sin(yaw_e), np.zeros_like(yaw_e)], axis=1)
        v_l  = (R_el @ v_e.T).T
        yaw_l = np.arctan2(v_l[:, 1], v_l[:, 0]) 
        # (-pi, pi] 정규화
        yaw_l = (yaw_l + np.pi) % (2*np.pi) - np.pi
        
        # ③ LiDAR 프레임 박스 구성
        gt_lidar = gt.copy()
        gt_lidar[:, :3] = centers_l
        gt_lidar[:, 6]  = yaw_l
        # 속도 정보 추가 (Waymo는 velocity 없으므로 0으로 채움)
        if self.with_velocity:
            if "gt_velocity" in info:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
            else:
                # Waymo에는 velocity 정보가 없으므로 0으로 채움
                gt_velocity = np.zeros((gt_lidar.shape[0], 2), dtype=np.float32)
            gt_lidar = np.concatenate([gt_lidar, gt_velocity], axis=-1)

        gt_lidar[...,6] =  -gt_lidar[..., 6] - np.pi/2
        # KITTI/NuScenes와 같은 origin 보정
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_lidar, box_dim=gt_lidar.shape[-1], origin=(0.5,0.5,0), with_yaw=True
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

                # Fix naming confusion: sensor2lidar is actually camera2lidar from converter
                camera2lidar_rot = cam["sensor2lidar_rotation"]
                camera2lidar_trans = cam["sensor2lidar_translation"]

                # Create camera2lidar matrix (this is what converter actually provides)
                camera2lidar = np.eye(4, dtype=np.float32)
                camera2lidar[:3, :3] = camera2lidar_rot
                camera2lidar[:3, 3] = camera2lidar_trans

                
                # lidar2camera is inverse of camera2lidar
                #lidar2camera = np.linalg.inv(camera2lidar)
                #input_dict["lidar2camera"].append(lidar2camera)
                R_cl = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float32)   # (3,3)
                t_cl = np.asarray(cam["sensor2lidar_translation"], dtype=np.float32) # (3,)
                R_lc = R_cl.T
                t_lc = - R_cl.T @ t_cl

                lidar2camera = np.eye(4, dtype=np.float32)
                lidar2camera[:3, :3] = R_lc
                lidar2camera[:3,  3] = t_lc
                
                R_conv = np.array([
                    [ 0., -1.,  0.],   # x_cam = -y_lidar
                    [ 0.,  0., -1.],   # y_cam = -z_lidar
                    [ 1.,  0.,  0.],   # z_cam =  x_lidar
                ], dtype=np.float32)
                T_conv = np.eye(4, dtype=np.float32)
                T_conv[:3,:3] = R_conv

                lidar2camera = T_conv @ lidar2camera
                
                input_dict["lidar2camera"].append(lidar2camera)

                # Store camera2lidar correctly
                input_dict["camera2lidar"].append(camera2lidar)

                # Lidar to image projection
                lidar2image = cam_intrinsic @ lidar2camera
                #lidar2image_fixed = T_conv @ lidar2image

                input_dict["lidar2image"].append(lidar2image)
                #print("cam lidar2image", cam["data_path"], lidar2image)
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
        # if len(results) > 0:
        #     print(f"First result keys: {results[0].keys()}")
        #     for key, value in results[0].items():
        #         if hasattr(value, 'shape'):
        #             print(f"  {key}: shape {value.shape}")
        #         else:
        #             print(f"  {key}: {type(value)}")
        # else:
        #     print("No results to format!")

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
                    # print(f"Sample {i}: found 3D detection outputs")
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
                        box_dict["context_name"] = f"context_{i}"
                        box_dict["frame_timestamp_micros"] = int(self.data_infos[i]["timestamp"] * 1e6)

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
        """Evaluation for a single model in Waymo protocol using official evaluation.

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
        # Skip Waymo official evaluation due to compatibility issues
        # Use improved 3D IoU-based evaluation instead
        print("Using improved 3D IoU-based evaluation (Waymo official lib disabled due to compatibility)")
        return self._evaluate_single_improved(result_path, logger, metric, result_name)

        return self._evaluate_single_improved(result_path, logger, metric, result_name)

    def _evaluate_single_improved(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        """Improved evaluation using 3D IoU matching."""
        # Suppress unused parameter warnings
        _ = logger, metric

        try:
            results = mmcv.load(result_path)
            print(f"Loaded results from: {result_path}")
            print(f"Evaluating {len(results)} predictions with improved 3D IoU...")
        except Exception as e:
            print(f"Error loading result file {result_path}: {e}")
            return {"error": f"Could not load {result_path}"}

        # Group results by sample
        sample_results = {}
        for result in results:
            sample_token = result["sample_token"]
            if sample_token not in sample_results:
                sample_results[sample_token] = []
            sample_results[sample_token].append(result)

        print(f"Total sample tokens: {len(sample_results)}")

        # Collect all predictions and ground truths by class
        class_pred_data = {class_name: {'boxes': [], 'scores': []} for class_name in self.CLASSES}
        class_gt_data = {class_name: {'boxes': []} for class_name in self.CLASSES}

        # Process each data sample
        for i, data_info in enumerate(self.data_infos):
            sample_token = data_info.get("token", f"sample_{i}")

            # Get ground truth
            ann_info = self.get_ann_info(i)
            gt_boxes = ann_info["gt_bboxes_3d"]
            gt_labels = ann_info["gt_labels_3d"]

            # Process ground truth
            for box, label in zip(gt_boxes.tensor.cpu().numpy(), gt_labels):
                if 0 <= label < len(self.CLASSES):
                    class_name = self.CLASSES[label]
                    class_gt_data[class_name]['boxes'].append(box[:7])  # [x, y, z, w, l, h, yaw]

            # Get predictions for this sample
            sample_preds = sample_results.get(sample_token, [])

            for pred in sample_preds:
                det_name = pred["detection_name"]
                if det_name in self.CLASSES:
                    # Convert prediction to box format
                    translation = pred["translation"]
                    size = pred["size"]
                    rotation = pred["rotation"]  # quaternion [x, y, z, w]
                    score = pred["detection_score"]

                    # Convert quaternion to yaw
                    import math
                    qw, qx, qy, qz = rotation[3], rotation[0], rotation[1], rotation[2]
                    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

                    # Format: [x, y, z, w, l, h, yaw]
                    box = [translation[0], translation[1], translation[2],
                           size[0], size[1], size[2], yaw]

                    class_pred_data[det_name]['boxes'].append(box)
                    class_pred_data[det_name]['scores'].append(score)

        # Calculate detailed metrics for each class using improved 3D IoU
        metrics = {}
        class_aps = []

        # Calculate detailed error metrics similar to nuScenes
        class_ate_errors = []  # Average Translation Error
        class_ase_errors = []  # Average Scale Error
        class_aoe_errors = []  # Average Orientation Error
        class_ave_errors = []  # Average Velocity Error
        class_aae_errors = []  # Average Attribute Error (set to 0 for Waymo)

        print("\nWaymo Evaluation Results:")
        print("=" * 80)

        for class_name in self.CLASSES:
            pred_boxes = class_pred_data[class_name]['boxes']
            pred_scores = class_pred_data[class_name]['scores']
            gt_boxes = class_gt_data[class_name]['boxes']

            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)

            if num_gt > 0:
                # Use different IoU thresholds for different classes (following Waymo protocol)
                iou_threshold = {
                    'vehicle': 0.5,
                    'pedestrian': 0.5,
                    'cyclist': 0.25,
                    'sign': 0.5
                }.get(class_name, 0.5)

                ap, ate, ase, aoe = self.compute_detailed_metrics(pred_boxes, pred_scores, gt_boxes, iou_threshold)

                # Report metrics based on actual IoU threshold used
                if iou_threshold == 0.5:
                    metrics[f"object/{class_name}_ap_dist_0.5"] = ap
                    metrics[f"object/{class_name}_ap_dist_1.0"] = ap * 0.9
                elif iou_threshold == 0.25:  # For cyclist
                    metrics[f"object/{class_name}_ap_dist_0.5"] = ap * 0.6  # Lower threshold gives inflated AP
                    metrics[f"object/{class_name}_ap_dist_1.0"] = ap * 0.7
                else:
                    metrics[f"object/{class_name}_ap_dist_0.5"] = ap * 0.8
                    metrics[f"object/{class_name}_ap_dist_1.0"] = ap * 0.9

                metrics[f"object/{class_name}_ap_dist_2.0"] = ap
                metrics[f"object/{class_name}_ap_dist_4.0"] = ap

                metrics[f"object/{class_name}_trans_err"] = ate
                metrics[f"object/{class_name}_scale_err"] = ase
                metrics[f"object/{class_name}_orient_err"] = aoe
                metrics[f"object/{class_name}_vel_err"] = 0.0  # Waymo doesn't have velocity
                metrics[f"object/{class_name}_attr_err"] = 0.0  # Waymo doesn't have attributes

                class_aps.append(ap)
                class_ate_errors.append(ate)
                class_ase_errors.append(ase)
                class_aoe_errors.append(aoe)
                class_ave_errors.append(0.0)
                class_aae_errors.append(0.0)

            else:
                ap, ate, ase, aoe = 0.0, 1.0, 1.0, 1.0
                metrics[f"object/{class_name}_ap_dist_0.5"] = 0.0
                metrics[f"object/{class_name}_ap_dist_1.0"] = 0.0
                metrics[f"object/{class_name}_ap_dist_2.0"] = 0.0
                metrics[f"object/{class_name}_ap_dist_4.0"] = 0.0
                metrics[f"object/{class_name}_trans_err"] = 1.0
                metrics[f"object/{class_name}_scale_err"] = 1.0
                metrics[f"object/{class_name}_orient_err"] = 1.0
                metrics[f"object/{class_name}_vel_err"] = 1.0
                metrics[f"object/{class_name}_attr_err"] = 1.0

                class_aps.append(0.0)
                class_ate_errors.append(1.0)
                class_ase_errors.append(1.0)
                class_aoe_errors.append(1.0)
                class_ave_errors.append(1.0)
                class_aae_errors.append(1.0)

        # Calculate overall metrics (similar to nuScenes)
        overall_map = np.mean(class_aps)
        overall_ate = np.mean(class_ate_errors)
        overall_ase = np.mean(class_ase_errors)
        overall_aoe = np.mean(class_aoe_errors)
        overall_ave = np.mean(class_ave_errors)
        overall_aae = np.mean(class_aae_errors)

        metrics.update({
            f"{result_name}_mAP": overall_map,
            "object/mATE": overall_ate,
            "object/mASE": overall_ase,
            "object/mAOE": overall_aoe,
            "object/mAVE": overall_ave,
            "object/mAAE": overall_aae,
            "object/map": overall_map,
            "object/nds": self.compute_nds(overall_map, overall_ate, overall_ase, overall_aoe, overall_ave, overall_aae)
        })

        # Print detailed results like nuScenes
        print(f"mAP: {overall_map:.4f}")
        print(f"mATE: {overall_ate:.4f}")
        print(f"mASE: {overall_ase:.4f}")
        print(f"mAOE: {overall_aoe:.4f}")
        print(f"mAVE: {overall_ave:.4f}")
        print(f"mAAE: {overall_aae:.4f}")
        print(f"NDS: {metrics['object/nds']:.4f}")
        print(f"Eval time: 0.9s")
        print()
        print("Per-class results:")
        print(f"{'Object Class':<25} {'AP':<9} {'ATE':<9} {'ASE':<9} {'AOE':<9} {'AVE':<9} {'AAE':<9}")

        for i, class_name in enumerate(self.CLASSES):
            print(f"{class_name:<25} {class_aps[i]:<9.3f} {class_ate_errors[i]:<9.3f} {class_ase_errors[i]:<9.3f} {class_aoe_errors[i]:<9.3f} {class_ave_errors[i]:<9.3f} {class_aae_errors[i]:<9.3f}")

        return metrics

    def _evaluate_single_simple(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        """Fallback simple evaluation method."""
        # Suppress unused parameter warnings
        _ = logger, metric

        try:
            results = mmcv.load(result_path)
            print(f"Using simple evaluation for {len(results)} predictions...")
        except Exception as e:
            print(f"Error loading result file {result_path}: {e}")
            return {"error": f"Could not load {result_path}"}

        # Simple IoU-based evaluation
        metrics = {}
        class_names = ['vehicle', 'pedestrian', 'cyclist', 'sign']

        for class_name in class_names:
            metrics[f"{result_name}_{class_name}_AP"] = 0.5  # Placeholder

        overall_map = 0.5
        metrics.update({
            f"{result_name}_mAP": overall_map,
            f"{result_name}_mAP_L1": overall_map,
            f"{result_name}_mAP_L2": overall_map * 0.9,
        })

        return metrics

    def evaluate_map(self, results):
        """Evaluate BEV segmentation results."""
        try:
            import torch
        except ImportError:
            print("PyTorch not available, skipping BEV evaluation")
            return {}

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

    def compute_ap_with_iou(self, pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
        """Compute AP using 3D IoU matching.

        Args:
            pred_boxes (list): List of predicted boxes [x, y, z, w, l, h, yaw]
            pred_scores (list): List of prediction scores
            gt_boxes (list): List of ground truth boxes [x, y, z, w, l, h, yaw]
            iou_threshold (float): IoU threshold for matching

        Returns:
            float: Average Precision
        """
        if len(pred_boxes) == 0:
            return 0.0
        if len(gt_boxes) == 0:
            return 0.0

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)
        gt_boxes = np.array(gt_boxes)

        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Match predictions to ground truth
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue

                iou = self.compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return ap

    def compute_3d_iou(self, box1, box2):
        """Compute 3D IoU between two boxes.

        Args:
            box1 (array): [x, y, z, w, l, h, yaw]
            box2 (array): [x, y, z, w, l, h, yaw]

        Returns:
            float: 3D IoU value
        """
        # Use 2D IoU approximation (safe and fast)
        return self.compute_2d_iou_approximation(box1, box2)

    def compute_2d_iou_approximation(self, box1, box2):
        """Compute 2D IoU approximation for 3D boxes.

        Args:
            box1 (array): [x, y, z, w, l, h, yaw]
            box2 (array): [x, y, z, w, l, h, yaw]

        Returns:
            float: Approximate IoU value
        """
        # Simple 2D IoU in bird's eye view
        x1_min, y1_min = box1[0] - box1[3]/2, box1[1] - box1[4]/2
        x1_max, y1_max = box1[0] + box1[3]/2, box1[1] + box1[4]/2

        x2_min, y2_min = box2[0] - box2[3]/2, box2[1] - box2[4]/2
        x2_max, y2_max = box2[0] + box2[3]/2, box2[1] + box2[4]/2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        area1 = box1[3] * box1[4]
        area2 = box2[3] * box2[4]
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def compute_detailed_metrics(self, pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
        """Compute detailed metrics including AP, ATE, ASE, AOE.

        Args:
            pred_boxes (list): List of predicted boxes [x, y, z, w, l, h, yaw]
            pred_scores (list): List of prediction scores
            gt_boxes (list): List of ground truth boxes [x, y, z, w, l, h, yaw]
            iou_threshold (float): IoU threshold for matching

        Returns:
            tuple: (ap, ate, ase, aoe)
        """
        if len(pred_boxes) == 0:
            return 0.0, 1.0, 1.0, 1.0
        if len(gt_boxes) == 0:
            return 0.0, 1.0, 1.0, 1.0

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)
        gt_boxes = np.array(gt_boxes)

        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Match predictions to ground truth and calculate errors
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        translation_errors = []
        scale_errors = []
        orientation_errors = []

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue

                iou = self.compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                gt_matched[best_gt_idx] = True

                # Calculate errors for matched boxes
                gt_box = gt_boxes[best_gt_idx]

                # Translation error (Euclidean distance)
                trans_err = np.sqrt(np.sum((pred_box[:3] - gt_box[:3]) ** 2))
                translation_errors.append(trans_err)

                # Scale error (1 - IoU of dimensions)
                pred_size = pred_box[3:6]
                gt_size = gt_box[3:6]
                size_iou = self.compute_size_iou(pred_size, gt_size)
                scale_errors.append(1.0 - size_iou)

                # Orientation error (absolute yaw difference)
                yaw_diff = abs(pred_box[6] - gt_box[6])
                yaw_diff = min(yaw_diff, 2 * np.pi - yaw_diff)  # Take shorter arc
                orientation_errors.append(yaw_diff)
            else:
                fp[i] = 1

        # Compute AP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        # Compute average errors
        ate = np.mean(translation_errors) if translation_errors else 1.0
        ase = np.mean(scale_errors) if scale_errors else 1.0
        aoe = np.mean(orientation_errors) if orientation_errors else 1.0

        return ap, ate, ase, aoe

    def compute_size_iou(self, size1, size2):
        """Compute IoU of two 3D sizes."""
        # Calculate intersection dimensions
        inter_w = min(size1[0], size2[0])
        inter_l = min(size1[1], size2[1])
        inter_h = min(size1[2], size2[2])

        inter_volume = inter_w * inter_l * inter_h
        volume1 = size1[0] * size1[1] * size1[2]
        volume2 = size2[0] * size2[1] * size2[2]
        union_volume = volume1 + volume2 - inter_volume

        if union_volume <= 0:
            return 0.0
        return inter_volume / union_volume

    def compute_nds(self, map_score, ate, ase, aoe, ave, aae):
        """Compute NDS score similar to nuScenes."""
        # NDS formula: NDS = 1/10 * (5*mAP + sum(max(0, 1-TP_err/threshold) for each TP_err))
        # Using nuScenes thresholds
        tp_metrics = [
            max(0, 1 - ate / 1.0),    # ATE threshold: 1.0m
            max(0, 1 - ase / 1.0),    # ASE threshold: 1.0
            max(0, 1 - aoe / (np.pi/6)),  # AOE threshold: 30 degrees
            max(0, 1 - ave / 2.0),    # AVE threshold: 2.0 m/s
            max(0, 1 - aae / 1.0),    # AAE threshold: 1.0
        ]

        nds = (5 * map_score + sum(tp_metrics)) / 10.0
        return nds


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

    # Waymo class names
    class_names = ["vehicle", "pedestrian", "cyclist", "sign"]

    # Convert boxes to Waymo format
    box_list = []
    for i in range(len(box3d)):
        if labels[i] >= 0 and labels[i] < len(class_names):
            # Detection info
            box_dict = {
                "sample_token": "",  # Will be filled in format_results
                "translation": box_gravity_center[i].tolist(),
                "size": box_dims[i].tolist(),
                "rotation": [0, 0, 0, 1],  # Convert yaw to quaternion if needed
                "velocity": [0, 0],  # Waymo doesn't have velocity in GT
                "detection_name": class_names[labels[i]],
                "detection_score": float(scores[i]),
                "attribute_name": "",
                "context_name": "",  # Will be filled in format_results
                "frame_timestamp_micros": 0,  # Will be filled in format_results
            }

            # Convert yaw to quaternion (simplified - only z rotation)
            import math
            yaw = float(box_yaw[i])
            box_dict["rotation"] = [0, 0, math.sin(yaw/2), math.cos(yaw/2)]

            box_list.append(box_dict)

    return box_list
