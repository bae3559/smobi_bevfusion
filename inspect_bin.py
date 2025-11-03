# inspect_bin.py
from waymo_open_dataset.metrics import metrics_pb2
from waymo_open_dataset import label_pb2
import numpy as np

def load_objs(path):
    objs = metrics_pb2.Objects()
    with open(path,'rb') as f:
        objs.ParseFromString(f.read())
    return objs.objects

def summarize(path, title):
    objs = load_objs(path)
    type_names = {
        label_pb2.Label.TYPE_VEHICLE: 'vehicle',
        label_pb2.Label.TYPE_PEDESTRIAN: 'ped',
        label_pb2.Label.TYPE_CYCLIST: 'cyclist',
        label_pb2.Label.TYPE_SIGN: 'sign',
        label_pb2.Label.TYPE_UNKNOWN: 'unknown',
    }
    by_type = {}
    for o in objs:
        t = o.object.type
        by_type.setdefault(t, []).append(o)
    print(f"\n== {title} ==")
    for t, arr in by_type.items():
        L = np.array([x.object.box.length for x in arr])
        W = np.array([x.object.box.width for x in arr])
        H = np.array([x.object.box.height for x in arr])
        S = np.array([getattr(x,'score',1.0) for x in arr])  # pred만 score 존재
        print(f"{type_names.get(t,t)}: n={len(arr)}  "
              f"L/W/H median=({np.median(L):.2f},{np.median(W):.2f},{np.median(H):.2f})  "
              f"score@p50={np.median(S):.3f}  score@p90={np.quantile(S,0.9):.3f}")

summarize("ground_truths.bin","GT")
summarize("bevfusion-pretrained-lidar-camerabackbone-5images.bin","PRED")
