#!/usr/bin/env python3
"""Check if sweeps are actually being loaded during visualization"""

import sys
sys.path.insert(0, '.')

import argparse
from mmcv import Config
from mmdet3d.datasets import build_dataset
import copy

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    # Load config
    from torchpack.utils.config import configs as torchpack_configs
    torchpack_configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(torchpack_configs), filename=args.config)

    # Build dataset with test pipeline
    dataset_cfg = cfg.data[args.split]
    dataset = build_dataset(dataset_cfg)

    print(f"Dataset type: {type(dataset).__name__}")
    print(f"Total samples: {len(dataset)}")
    print()

    # Check first few samples
    for i in range(min(3, len(dataset))):
        print(f"=== Sample {i} ===")

        # Get raw info (before pipeline)
        info = dataset.data_infos[i]
        print(f"Info keys: {list(info.keys())}")

        if 'sweeps' in info:
            print(f"Number of sweeps in info: {len(info['sweeps'])}")
            if len(info['sweeps']) > 0:
                print(f"First sweep: {info['sweeps'][0]}")
        else:
            print("NO SWEEPS in info!")

        # Get processed data (after pipeline)
        data = dataset[i]
        if 'points' in data:
            points = data['points']._data
            print(f"Points shape after pipeline: {points.shape}")
            print(f"Points dtype: {points.dtype}")

            # Check time_lag (5th dimension)
            if points.shape[1] >= 5:
                time_lags = points[:, 4]
                unique_lags = len(set(time_lags.tolist()))
                print(f"Unique time_lag values: {unique_lags}")
                print(f"Time_lag range: [{time_lags.min():.3f}, {time_lags.max():.3f}]")

                if unique_lags > 1:
                    print("✓ Multiple sweeps detected!")
                else:
                    print("✗ Only single frame (no sweeps loaded)")
        print()

if __name__ == "__main__":
    main()
