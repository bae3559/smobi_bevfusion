import os
import mmcv
import tensorflow as tf
import numpy as np
from pathlib import Path
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# Waymo object type → class name 매핑
WAYMO_CLASSES = {
    1: "vehicle",
    2: "pedestrian",
    3: "sign",
    4: "cyclist"
}


def parse_tfrecord(tfrecord_path, out_dir):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    infos = []

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_idx = f"{frame.context.name}_{frame.timestamp_micros}"

        ## frame.context.laser_calibrations  -> extrinsic, intrinsic
        ##### 1. LiDAR Extrinsic (TOP LiDAR → EGO) #####
        top_laser = next(l for l in frame.context.laser_calibrations
                         if l.name == open_dataset.LaserName.TOP)
        lidar2ego = np.array(top_laser.extrinsic.transform, dtype=np.float32).reshape(4, 4)

        ##### 2. LiDAR Point Cloud 저장 (ego 기준) #####
        from waymo_open_dataset.utils import range_image_utils, transform_utils

        (range_images, camera_projections, seg_labels, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # Use official method to extract point cloud (already in ego coordinates)
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points_list = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)

        for ri_index in [0, 1]:  # Process both returns
            for c in calibrations:
                range_image = range_images[c.name][ri_index]
                if len(c.beam_inclinations) == 0:
                    beam_inclinations = range_image_utils.compute_inclination(
                        tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                        height=range_image.shape.dims[0])
                else:
                    beam_inclinations = tf.constant(c.beam_inclinations)

                beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

                range_image_tensor = tf.reshape(
                    tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
                pixel_pose_local = None
                frame_pose_local = None
                if c.name == open_dataset.LaserName.TOP:
                    pixel_pose_local = range_image_top_pose_tensor
                    pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                    frame_pose_local = tf.expand_dims(frame_pose, axis=0)

                range_image_mask = range_image_tensor[..., 0] > 0
                range_image_cartesian = \
                    range_image_utils.extract_point_cloud_from_range_image(
                        tf.expand_dims(range_image_tensor[..., 0], axis=0),
                        tf.expand_dims(extrinsic, axis=0),
                        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                        pixel_pose=pixel_pose_local,
                        frame_pose=frame_pose_local)

                mask_index = tf.where(range_image_mask)
                range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
                points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

                if points_tensor.shape[0] > 0:
                    # Add intensity and elongation
                    intensity = tf.gather_nd(range_image_tensor[..., 1], mask_index)
                    elongation = tf.gather_nd(range_image_tensor[..., 2], mask_index)
                    # Combine [x, y, z, intensity, elongation, mask_index]
                    points_with_features = tf.concat([
                        points_tensor,
                        tf.expand_dims(intensity, 1),
                        tf.expand_dims(elongation, 1)
                    ], axis=1)
                    points_list.append(points_with_features.numpy())


        if points_list:
            points_all = np.concatenate(points_list, axis=0)
            #print(f"DEBUG: Points after official extraction (ego coords) - X: [{points_all[:, 0].min():.2f}, {points_all[:, 0].max():.2f}], Y: [{points_all[:, 1].min():.2f}, {points_all[:, 1].max():.2f}], Z: [{points_all[:, 2].min():.2f}, {points_all[:, 2].max():.2f}]")
            #print(f"DEBUG: Total points: {points_all.shape[0]}")
        else:
            points_all = np.empty((0, 5), dtype=np.float32)
            #print("DEBUG: No valid points found!")

        # lidar2ego: TOP LiDAR → EGO (이미 위에서 구해둠)
        T_le = lidar2ego.astype(np.float32)
        T_el = np.linalg.inv(T_le).astype(np.float32)  # EGO → LiDAR

        # xyz만 변환 (EGO→LiDAR)
        N = points_all.shape[0]
        if N > 0:
            xyz1 = np.concatenate([points_all[:, :3], np.ones((N, 1), np.float32)], axis=1)  # (N,4)
            xyz_l = (T_el @ xyz1.T).T[:, :3]
            points_all[:, :3] = xyz_l



        # range_image_cartesian (N,3) : (x,y,z)
        # intensity (N,)
        # elongation (N, )

        # Debug: print shape information
        #print(f"DEBUG - Total points shapes: {[p.shape for p in points]}")
        #print(f"DEBUG - Concatenated shape: {points_all.shape}")
        #print(f"DEBUG - Saving to: {frame_idx}.bin")

        lidar_out = Path(out_dir) / "lidar"
        lidar_out.mkdir(parents=True, exist_ok=True)
        lidar_path = lidar_out / f"{frame_idx}.bin"

        points_all.astype(np.float32).tofile(lidar_path)

        #---------------------- 2. 카메라 저장----------------------
        cams = {}

        # NuScenes 6-camera structure 매핑
        # Waymo: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT (5개)
        # NuScenes: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT (6개)
        waymo_to_nuscenes_mapping = {
            'FRONT': 'CAM_FRONT',
            'FRONT_LEFT': 'CAM_FRONT_LEFT',
            'FRONT_RIGHT': 'CAM_FRONT_RIGHT',
            'SIDE_LEFT': 'CAM_BACK_LEFT',   
            'SIDE_RIGHT': 'CAM_BACK_RIGHT'  
        }

        for img in frame.images:
            cam_name = open_dataset.CameraName.Name.Name(img.name).upper()
            nuscenes_cam_name = waymo_to_nuscenes_mapping.get(cam_name, f"CAM_{cam_name}")

            img_dir = Path(out_dir) / "images" / nuscenes_cam_name
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{frame_idx}.jpg"
            with open(img_path, "wb") as f:
                f.write(img.image)

            # Calibration
            calib = next((c for c in frame.context.camera_calibrations if c.name == img.name), None)
            #print("calibartion",calib)
            if calib is not None:
                # Waymo intrinsic parameters: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
                # We only need the first 4 for the camera matrix
                intrinsic_params = np.array(calib.intrinsic, dtype=np.float32)
                #print('camera_intrinsic', intrinsic_params)
                fx, fy, cx, cy = intrinsic_params[0], intrinsic_params[1], intrinsic_params[2], intrinsic_params[3]
                intrinsics = np.array([[fx,  0, cx],
                                     [ 0, fy, cy],
                                     [ 0,  0,  1]], dtype=np.float32)
                # extrinsic : camera to vehicle
                extrinsic = np.reshape(np.array(calib.extrinsic.transform), [4, 4])
            else:
                intrinsics = np.eye(3)
                extrinsic = np.eye(4)

            # ✅ CAM→EGO (extrinsic 그대로 사용)
            sensor2ego_rotation = extrinsic[:3, :3]
            sensor2ego_translation = extrinsic[:3, 3]

            # ✅ CAM→LiDAR = (lidar2ego^-1) @ (CAM→EGO)
            sensor2lidar = np.linalg.inv(lidar2ego) @ extrinsic
            sensor2lidar_rotation = sensor2lidar[:3, :3]
            sensor2lidar_translation = sensor2lidar[:3, 3]

            cams[nuscenes_cam_name] = dict(
                data_path=str(img_path),
                cam_intrinsic=intrinsics,
                sensor2ego_rotation=sensor2ego_rotation,
                sensor2ego_translation=sensor2ego_translation,
                sensor2lidar_rotation=sensor2lidar_rotation,
                sensor2lidar_translation=sensor2lidar_translation,
                timestamp=frame.timestamp_micros,
            )

        # CAM_BACK 더미 카메라 추가 (Waymo에는 후방 카메라가 없음)
        if 'CAM_BACK' not in cams:
            # 더미 이미지 생성 (검은색 이미지)
            dummy_img_dir = Path(out_dir) / "images" / "CAM_BACK"
            dummy_img_dir.mkdir(parents=True, exist_ok=True)
            dummy_img_path = dummy_img_dir / f"{frame_idx}.jpg"

            # 검은색 더미 이미지 생성 (Waymo 카메라와 같은 크기)
            from PIL import Image as PILImage
            dummy_img = PILImage.fromarray(np.zeros((1280, 1920, 3), dtype=np.uint8))  # Waymo 카메라 해상도
            dummy_img.save(dummy_img_path)

            # 더미 calibration (후방 카메라 위치로 가정)
            # 다른 카메라와 비슷한 intrinsic 사용
            dummy_intrinsics = np.array([
                [2050.0, 0.0, 960.0],   # 다른 카메라와 비슷한 focal length, center
                [0.0, 2050.0, 640.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            # 후방 카메라 위치 (ego vehicle 뒤쪽, 180도 회전)
            dummy_extrinsic = np.array([
                [-1.0, 0.0, 0.0, -1.5],  # 뒤쪽 1.5m, 180도 회전 (front를 향함)
                [0.0, -1.0, 0.0, 0.0],   # Y축 반전
                [0.0, 0.0, 1.0, 2.1],    # 높이는 다른 카메라와 동일
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

            # ✅ Compute proper sensor2lidar for dummy camera
            dummy_sensor2lidar = np.linalg.inv(lidar2ego) @ dummy_extrinsic
            dummy_sensor2lidar_rotation = dummy_sensor2lidar[:3, :3]
            dummy_sensor2lidar_translation = dummy_sensor2lidar[:3, 3]

            cams['CAM_BACK'] = dict(
                data_path=str(dummy_img_path),
                cam_intrinsic=dummy_intrinsics,
                sensor2ego_rotation=dummy_extrinsic[:3, :3],
                sensor2ego_translation=dummy_extrinsic[:3, 3],
                sensor2lidar_rotation=dummy_sensor2lidar_rotation,
                sensor2lidar_translation=dummy_sensor2lidar_translation,
                timestamp=frame.timestamp_micros,
            )

        #---------------------- 3. 어노테이션 ----------------------
        ### waymo는 gt_velocity가 제공되지 않는다.
        ### vx = 
        gt_boxes, gt_names, num_pts = [], [], []

        # L->E (lidar2ego)
        R_le = lidar2ego[:3, :3].astype(np.float32)
        t_le = lidar2ego[:3, 3].astype(np.float32)
        T_le = np.eye(4, dtype=np.float32)
        T_le[:3, :3] = R_le
        T_le[:3, 3]  = t_le

        # E->L
        T_el = np.linalg.inv(T_le).astype(np.float32)
        R_el = T_el[:3, :3]  # = R_le.T
        for label in frame.laser_labels:
            cls_name = WAYMO_CLASSES.get(label.type, "unknown")

            # EGO 기준 중심/치수/yaw (Waymo 정의 그대로)
            cx_e = float(label.box.center_x)
            cy_e = float(label.box.center_y)
            cz_e = float(label.box.center_z)

            dx = float(label.box.width)  # [length, width, height]
            dy = float(label.box.length)
            dz = float(label.box.height)

            yaw_e = float(label.box.heading)

            # --- 중심: EGO -> LiDAR ---
            p_e = np.array([cx_e, cy_e, cz_e, 1.0], dtype=np.float32)
            p_l = (T_el @ p_e)[:3]  # (3,)

            # --- yaw: EGO -> LiDAR (방향 벡터 회전) ---
            v_e = np.array([np.cos(yaw_e), np.sin(yaw_e), 0.0], dtype=np.float32)  # EGO 평면 방향
            v_l = (R_el @ v_e)  # LiDAR 평면 방향
            yaw_l = np.arctan2(v_l[1], v_l[0])
            # [-pi, pi) 정규화
            yaw_l = (yaw_l + np.pi) % (2 * np.pi) - np.pi

            # LiDAR 기준 박스: [x, y, z, dx, dy, dz, yaw]
            box_l = [p_l[0], p_l[1], p_l[2], dx, dy, dz, -yaw_l-np.pi/2]
            gt_boxes.append(box_l)
            gt_names.append(cls_name)
            num_pts.append(label.num_lidar_points_in_box)

        #----------------------4. Pose (ego → global) ----------------------
        pose_matrix = np.array(frame.pose.transform).reshape(4, 4)
        ego2global_translation = pose_matrix[:3, 3]
        ego2global_rotation = pose_matrix[:3, :3]

        #------------------5. info dict ----------------------
        info = dict(
            token=frame_idx,
            lidar_path=str(lidar_path),
            sweeps=[],
            cams=cams,
            timestamp=frame.timestamp_micros,
            gt_boxes=gt_boxes,
            gt_names=np.array(gt_names),
            num_lidar_pts=np.array(num_pts, dtype=np.int32),
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego[:3, 3],
            lidar2ego_rotation=lidar2ego[:3, :3],
        )
        infos.append(info)

    return infos

def create_waymo_infos(root_path, out_dir, version, extra_tag="waymo"):
    if version == "trainval":
        train_tfrecords = sorted(Path(root_path , "/training").rglob("*.tfrecord"))
        val_tfrecords = sorted(Path(root_path , "/validatiaon").rglob("*.tfrecord"))
        train_infos = []
        for tfrecord in train_tfrecords:
            train_infos.extend(parse_tfrecord(str(tfrecord), out_dir))
        out_file_train = Path(out_dir) / f"{extra_tag}_infos_train.pkl"
        mmcv.dump(dict(infos=train_infos, metadata=dict(version=version)), str(out_file_train))
        print(f"Saved {len(train_infos)} frames to {out_file_train}")
        val_infos = []
        for tfrecord in val_tfrecords:
            val_infos.extend(parse_tfrecord(str(tfrecord), out_dir))
        out_file_val = Path(out_dir) / f"{extra_tag}_infos_val.pkl"
        mmcv.dump(dict(infos=val_infos, metadata=dict(version=version)), str(out_file_val))
        print(f"Saved {len(val_infos)} frames to {out_file_val}")
    elif version== "test":
        test_tfrecords = sorted(Path(root_path , "/test").rglob("*.tfrecord"))
        test_infos = []
        for tfrecord in test_tfrecords:
            test_infos.extend(parse_tfrecord(str(tfrecord), out_dir))
        out_file_val = Path(out_dir) / f"{extra_tag}_infos_val.pkl"
        mmcv.dump(dict(infos=test_infos, metadata=dict(version=version)), str(out_file_val))
        print(f"Saved {len(test_infos)} frames to {out_file_val}")



def create_waymo_infos_mini(root_path, out_dir, version="mini", extra_tag="waymo"):
    train_tfrecords = sorted(Path(root_path, "waymo_format/training").rglob("*.tfrecord"))
    val_tfrecords = sorted(Path(root_path, "waymo_format/validation").rglob("*.tfrecord"))

    train_infos = []
    for tfrecord in train_tfrecords:
        train_infos.extend(parse_tfrecord(str(tfrecord), out_dir))
    out_file_train = Path(out_dir) / f"{extra_tag}_infos_train.pkl"
    mmcv.dump(dict(infos=train_infos, metadata=dict(version=version)), str(out_file_train))
    print(f"Saved {len(train_infos)} frames to {out_file_train}")

    val_infos = []
    for tfrecord in val_tfrecords:
        val_infos.extend(parse_tfrecord(str(tfrecord), out_dir))
    out_file_val = Path(out_dir) / f"{extra_tag}_infos_val.pkl"
    mmcv.dump(dict(infos=val_infos, metadata=dict(version=version)), str(out_file_val))
    print(f"Saved {len(val_infos)} frames to {out_file_val}")
