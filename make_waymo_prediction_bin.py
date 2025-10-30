#!/usr/bin/env python
"""
    mmedet3d/BEVFusion 결과(pkl/json) + waymo_infos_val.pkl
    --> waymo 공식 metric 툴에서 쓰는 prediction.bin으로 변환

    사용예:
    python make_waymo_prediction_bin.py \
        --infos data/waymo/waymo_infos_val.pkl \ 
        --results work_dirs/waymo/val_results.pkl \
        --out prediction.bin
"""


import argparse
import pickle
import json
import os

import numpy as np

from waymo_open_dataset.metrics import metrics_pb2
from waymo_open_dataset import label_pb2

def load_infos(path):
    with open(path, "rb") as f:
        infos = pickle.load(f)
    return infos


def load_results(path):
    if path.endswith(".pkl"): 
        with open(path, "rb") as f:
            results = pickle.load(f)

    elif path.endswith(".json"):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unknown results format: {path}")
    return results


### 확인해보고 맞는 info name 쓰기
def get_ctx_and_ts(info):
    ctx = ( 
        info.get("context_name")
    )
    ts = (
        info.get("timestamp")
    )
    return ctx, int(ts)

# 모델이 0:vehicle, 1:pedestrian, 2:sign, 3:cyclist 
MODEL_TO_WAYMO = {
    0: label_pb2.Label.TYPE_VEHICLE,    # 1
    1: label_pb2.Label.TYPE_PEDESTRIAN, # 2
    2: label_pb2.Label.TYPE_SIGN,       # 3
    3: label_pb2.Label.TYPE_CYCLIST,    # 4
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infos", required=True, help ="waymo_infos_val.pkl 경로")
    parser.add_argument("--results", required=True, help="mmdet3d 추론 결과 pkl/json")
    parser.add_argument("--out", default="prediction.bin", help="저장할 bin 파일 이름")
    args = parser.parse_args()

    infos = load_infos(args.infos)
    results = load_results(args.result)

    # results가 dict가 아니라 list여야 Len이 맞음
    assert len(results) == len(infos), \
        f"results ({len(results)}) and infos ({len(infos)}) must have same length"
    
    objects = metrics_pb2.Objects()

    for idx, det in enumerate(results):
        info = infos[idx]
        ctx, ts = get_ctx_and_ts(info)

        # mmdet3d 결과 (형식 다시 확인해보기)
        boxes_3d = det["boxes_3d"].tensor.numpy()
        scores_3d = det["scores_3d"].numpy()
        labels_3d = det["labels_3d"].numpy()

        for box, score, label in zip(boxes_3d, scores_3d, labels_3d):
            o = metrics_pb2.Object()
            o.context_name = ctx
            o.frame_timestamp_micros = ts

            # box: [x, y, z, dx, dy, dz, yaw]
            b = label_pb2.Label.Box()
            b.center_x = float(box[0])
            b.center_y = float(box[1])
            b.center_z = float(box[2])
            b.length = float(box[3])
            b.width = float(box[4])
            b.height = float(box[5])
            b.heading = float(box[6])

            o.object.box.CopyFrom(b)

            # 모델 라벨 -> waymo 라벨 매핑
            waymo_type = MODEL_TO_WAYMO.get(int(label), label_pb2.Label.TYPE_KNOWN)
            o.object.type = waymo_type

            o.score = float(score)

            objects.objects.append(o)

    with open(args.out, "wb") as f:
        f.write(objects.SerializeToString())

    print(f"[OK] wrote {len(objects.objects)} objects to {args.out}")

if __name__ == "__main__":
    main()