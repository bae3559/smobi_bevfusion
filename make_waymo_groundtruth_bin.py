#!/usr/bin/env python
import argparse, pickle
from typing import Any, List, Tuple
import numpy as np

from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2


NAME2WAYMO = {
    "vehicle":  label_pb2.Label.TYPE_VEHICLE,
    "car":      label_pb2.Label.TYPE_VEHICLE,
    "pedestrian": label_pb2.Label.TYPE_PEDESTRIAN,
    "person":     label_pb2.Label.TYPE_PEDESTRIAN,
    "cyclist":  label_pb2.Label.TYPE_CYCLIST,
    "bicycle":  label_pb2.Label.TYPE_CYCLIST,
    "sign":     label_pb2.Label.TYPE_SIGN,
    "traffic_sign": label_pb2.Label.TYPE_SIGN,
}

def normalize_infos(obj: Any) -> List[dict]:
    if isinstance(obj, dict) and "infos" in obj and isinstance(obj["infos"], list):
        return obj["infos"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported infos structure: {type(obj)}")

def load_infos(path: str) -> List[dict]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return normalize_infos(raw)

def parse_ctx_ts_from_token(token: str) -> Tuple[str, int]:
    ctx = token.rsplit("_", 1)[0]
    ts = int(token.rsplit("_", 1)[1])
    return ctx, ts

def lidar_to_vehicle_centers_and_yaw(xyz_l: np.ndarray, yaw_l: np.ndarray,
                                     R: np.ndarray, t: np.ndarray):
    use_transform = False
    if use_transform:
        xyz_v = xyz_l @ R.T + t[None, :]
        # z-축 회전 성분만 반영 (일반적 가정)
        yaw_R = np.arctan2(R[1, 0], R[0, 0])
        yaw_v = yaw_l + yaw_R
    else:
        xyz_v = xyz_l
        yaw_v = yaw_l
    return xyz_v, yaw_v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="waymo_infos_val.pkl")
    ap.add_argument("--out", default="ground_truths.bin")
    args = ap.parse_args()

    infos = load_infos(args.infos)
    objects = metrics_pb2.Objects()

    for info in infos:
        token = info["token"]
        ctx, ts = parse_ctx_ts_from_token(token)

        R = np.asarray(info["lidar2ego_rotation"], dtype=np.float32)      # (3,3)
        t = np.asarray(info["lidar2ego_translation"], dtype=np.float32)   # (3,)

        boxes = np.asarray(info.get("gt_boxes", []))        # (N,7): x,y,z,dx,dy,dz,yaw in LiDAR
        names = np.asarray(info.get("gt_names", []))
        numpts = np.asarray(info.get("num_lidar_pts", []))

        if boxes.size == 0:
            continue

        xyz_l = boxes[:, :3]
        yaw_l = boxes[:, 6]
        xyz_v, yaw_v = lidar_to_vehicle_centers_and_yaw(xyz_l, yaw_l, R, t)

        length = boxes[:, 3]
        width  = boxes[:, 4]
        height = boxes[:, 5]

        for i in range(boxes.shape[0]):
            o = metrics_pb2.Object()
            o.context_name = ctx
            o.frame_timestamp_micros = int(ts)

            b = label_pb2.Label.Box()
            b.center_x, b.center_y, b.center_z = map(float, xyz_v[i])
            b.length, b.width, b.height = float(length[i]), float(width[i]), float(height[i])
            b.heading = float(yaw_v[i])
            o.object.box.CopyFrom(b)

            cls = str(names[i]).lower() if i < len(names) else "vehicle"
            o.object.type = NAME2WAYMO.get(cls, label_pb2.Label.TYPE_UNKNOWN)

            if i < len(numpts):
                o.object.num_lidar_points_in_box = int(numpts[i])
            o.object.detection_difficulty_level = 0  # 필요시 커스텀

            objects.objects.append(o)

    with open(args.out, "wb") as f:
        f.write(objects.SerializeToString())
    print(f"[OK] wrote", len(objects.objects), "GT objects to", args.out)

if __name__ == "__main__":
    main()
