#!/usr/bin/env python3
"""
mmdet3d/BEVFusion 추론 결과(pkl/json) + waymo_infos_val.pkl
→ Waymo 공식 metric 툴에서 쓰는 prediction.bin 생성 (좌표변환 없음)

사용 예:
python make_waymo_prediction_bin.py \
  --infos data/waymo/Waymo_mini/waymo_infos_val.pkl \
  --results runs/251031/bevfusion-pretrained-lidar-camerabackbone-5images/val_results.pkl \
  --out bevfusion-pretrained-lidar-camerabackbone-5images.bin
"""

import argparse
import pickle
import json
import os
import numpy as np

from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2


# 네 구성(0: vehicle, 1: pedestrian, 2: cyclist, 3: sign)에 맞춘 매핑
MODEL_TO_WAYMO = {
    0: label_pb2.Label.TYPE_VEHICLE,
    1: label_pb2.Label.TYPE_PEDESTRIAN,
    2: label_pb2.Label.TYPE_CYCLIST,
    3: label_pb2.Label.TYPE_SIGN,
}

def load_infos(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # pkl 구조: {'infos': [...], 'metadata': {...}} 혹은 바로 list
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported infos format: {type(data)}")

def load_results(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            results = pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unknown results format: {path}")
    # 일부 포맷은 dict에 'results'나 'pts_bbox'로 감싸져 있을 수 있음 → 리스트로 평탄화
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    assert isinstance(results, list), f"results must be a list, got {type(results)}"
    return results

def get_ctx_and_ts(info):
    """
    좌표변환 없이 프레임 페어링만 맞추기 위해 context_name/ts를 추출.
    - token 형식: '{context}_{timestamp_micros}'
    - timestamp는 info['timestamp']가 이미 microseconds인 케이스가 많음
    """
    token = info.get("token", "")
    ts_from_info = info.get("timestamp", None)

    context_name = None
    frame_ts = None

    if token and "_" in token:
        # 맨 뒤의 _마이크로초 타임스탬프 분리
        try:
            context_name, ts_str = token.rsplit("_", 1)
            frame_ts = int(ts_str)
        except Exception:
            context_name = None
            frame_ts = None

    if context_name is None:
        # fallback: context_name 필드가 있으면 사용
        context_name = info.get("context_name", None)
        # 그래도 없으면 lidar_path의 상위 폴더명 정도로 대체(프레임 수집이 같으면 매칭 가능)
        if context_name is None:
            lidar_path = info.get("lidar_path", "")
            context_name = os.path.basename(os.path.dirname(lidar_path)) or "unknown_context"

    if frame_ts is None:
        # info['timestamp']가 이미 μs면 그대로, s(부동소수)면 1e6 곱
        if isinstance(ts_from_info, (int, np.integer)):
            frame_ts = int(ts_from_info)
        elif isinstance(ts_from_info, float):
            # seconds → micros 로 가정
            frame_ts = int(round(ts_from_info * 1e6))
        else:
            # 마지막 fallback: 0
            frame_ts = 0

    return context_name, int(frame_ts)

def extract_boxes_scores_labels(one_result):
    """
    mmdet3d 결과 한 샘플에서 boxes/scores/labels를 뽑아 nx7, n, n 형태로 반환.
    - 포맷 다양성에 대응: {'boxes_3d','scores_3d','labels_3d'} 혹은 {'pts_bbox': {...}}
    - boxes_3d가 LiDARInstance3DBoxes인 경우 .tensor 또는 (gravity_center,dims,yaw) 사용
    """
    det = one_result
    if isinstance(det, dict) and "pts_bbox" in det:
        det = det["pts_bbox"]

    if not isinstance(det, dict):
        raise ValueError(f"Unexpected result entry type: {type(det)}")

    boxes_3d = det.get("boxes_3d", None)
    scores_3d = det.get("scores_3d", None)
    labels_3d = det.get("labels_3d", None)

    if boxes_3d is None or scores_3d is None or labels_3d is None:
        # 비어있는 결과(감지 없음)일 수 있음 → 빈 배열 반환
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    # boxes_3d가 객체(LiDARInstance3DBoxes)인 경우 처리
    # 가능 필드:
    # - .tensor (N,7): [x,y,z,dx,dy,dz,yaw]
    # - .gravity_center (N,3), .dims (N,3), .yaw (N,)
    if hasattr(boxes_3d, "tensor"):
        b = boxes_3d.tensor
        boxes = b.detach().cpu().numpy() if hasattr(b, "detach") else b.numpy()
    elif (hasattr(boxes_3d, "gravity_center") and
          hasattr(boxes_3d, "dims") and
          hasattr(boxes_3d, "yaw")):
        gc = boxes_3d.gravity_center
        dm = boxes_3d.dims
        yw = boxes_3d.yaw
        gc = gc.detach().cpu().numpy() if hasattr(gc, "detach") else gc.numpy()
        dm = dm.detach().cpu().numpy() if hasattr(dm, "detach") else dm.numpy()
        yw = yw.detach().cpu().numpy() if hasattr(yw, "detach") else yw.numpy()
        # mmdet3d의 기본 형태에 맞춰 [x,y,z, dx,dy,dz, yaw]
        boxes = np.concatenate([gc, dm, yw[:, None]], axis=1).astype(np.float32)
    else:
        # 이미 numpy일 수도 있음
        boxes = np.asarray(boxes_3d, dtype=np.float32)

    scores = scores_3d.detach().cpu().numpy() if hasattr(scores_3d, "detach") else np.asarray(scores_3d)
    labels = labels_3d.detach().cpu().numpy() if hasattr(labels_3d, "detach") else np.asarray(labels_3d)

    # 안전 캐스팅
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32).reshape(-1)
    labels = labels.astype(np.int64).reshape(-1)

    # 박스가 (N,7)이 아니라면 방어 (예: code_size=10인 경우 앞 7개만 사용)
    if boxes.shape[1] < 7:
        raise ValueError(f"boxes_3d has shape {boxes.shape}, expected at least 7 dims")
    if boxes.shape[1] > 7:
        boxes = boxes[:, :7].copy()

    return boxes, scores, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infos", required=True, help="waymo_infos_val.pkl 경로")
    parser.add_argument("--results", required=True, help="mmdet3d 추론 결과 pkl/json 경로")
    parser.add_argument("--out", default="prediction.bin", help="저장할 bin 파일 이름")
    args = parser.parse_args()

    infos = load_infos(args.infos)
    results = load_results(args.results)
    assert len(results) == len(infos), f"results ({len(results)}) vs infos ({len(infos)}) 길이 불일치"

    objects = metrics_pb2.Objects()
    total = 0

    for i, det in enumerate(results):
        # 컨텍스트/타임스탬프 (좌표변환과 무관)
        info = infos[i]
        ctx, ts = get_ctx_and_ts(info)

        # 예측 박스/스코어/라벨 추출 (그대로 사용: 좌표/heading 변환 안 함)
        boxes, scores, labels = extract_boxes_scores_labels(det)
        if boxes.size == 0:
            continue

        for b, sc, lb in zip(boxes, scores, labels):
            # b: [x, y, z, dx, dy, dz, yaw]  (LiDAR 그대로)
            x, y, z, dx, dy, dz, yaw = map(float, b.tolist())

            o = metrics_pb2.Object()
            o.context_name = str(ctx)
            o.frame_timestamp_micros = int(ts)

            # Waymo Label.Box: length/width/height는 그대로 복사
            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length  = dx
            box.width   = dy
            box.height  = dz
            box.heading = yaw 

            o.object.box.CopyFrom(box)

            # 클래스 매핑 (0:veh, 1:ped, 2:cyc, 3:sign)
            o.object.type = MODEL_TO_WAYMO.get(int(lb), label_pb2.Label.TYPE_UNKNOWN)

            # 스코어
            o.score = float(sc)

            objects.objects.append(o)
            total += 1

    with open(args.out, "wb") as f:
        f.write(objects.SerializeToString())

    print(f"[OK] wrote {total} objects to {args.out}")

if __name__ == "__main__":
    main()
