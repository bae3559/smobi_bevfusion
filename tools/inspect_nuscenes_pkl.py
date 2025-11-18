#!/usr/bin/env python3
"""Inspect NuScenes pickle files to see their structure"""

import pickle
import numpy as np

def inspect_infos(pkl_path):
    print(f"\n{'='*80}")
    print(f"File: {pkl_path}")
    print(f"{'='*80}\n")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Top-level keys: {list(data.keys())}")
    print()

    if 'metadata' in data:
        print(f"Metadata: {data['metadata']}")
        print()

    if 'infos' in data:
        infos = data['infos']
        print(f"Number of samples: {len(infos)}")
        print()

        if len(infos) > 0:
            print("=" * 80)
            print("First sample structure:")
            print("=" * 80)
            sample = infos[0]

            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], dict):
                            print(f"  {key}: list of {len(value)} dicts")
                            print(f"    First dict keys: {list(value[0].keys())}")
                        elif isinstance(value[0], np.ndarray):
                            print(f"  {key}: list of {len(value)} ndarrays, first shape={value[0].shape}")
                        else:
                            print(f"  {key}: list of {len(value)} {type(value[0]).__name__}")
                    else:
                        print(f"  {key}: empty list")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys: {list(value.keys())}")
                    # Show one camera example
                    if len(value) > 0:
                        first_cam = list(value.keys())[0]
                        cam_info = value[first_cam]
                        if isinstance(cam_info, dict):
                            print(f"    Example ({first_cam}): {list(cam_info.keys())}")
                        elif isinstance(cam_info, list) and len(cam_info) > 0:
                            print(f"    Example ({first_cam}): list of {len(cam_info)} items")
                            if isinstance(cam_info[0], dict):
                                print(f"      First item keys: {list(cam_info[0].keys())}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value if not isinstance(value, (list, dict)) else '...'}")

            print()
            print("=" * 80)
            print("Sample with annotations (if available):")
            print("=" * 80)

            # Find a sample with annotations
            sample_with_annos = None
            for s in infos[:10]:
                if 'gt_boxes' in s and len(s.get('gt_boxes', [])) > 0:
                    sample_with_annos = s
                    break

            if sample_with_annos:
                if 'gt_boxes' in sample_with_annos:
                    print(f"  gt_boxes: shape={sample_with_annos['gt_boxes'].shape if isinstance(sample_with_annos['gt_boxes'], np.ndarray) else len(sample_with_annos['gt_boxes'])}")
                if 'gt_names' in sample_with_annos:
                    print(f"  gt_names: {sample_with_annos['gt_names'][:5]}...")
                if 'num_lidar_pts' in sample_with_annos:
                    print(f"  num_lidar_pts: {sample_with_annos['num_lidar_pts'][:5]}...")
                if 'num_radar_pts' in sample_with_annos:
                    print(f"  num_radar_pts: {sample_with_annos['num_radar_pts'][:5]}...")

def inspect_dbinfos(pkl_path):
    print(f"\n{'='*80}")
    print(f"File: {pkl_path}")
    print(f"{'='*80}\n")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        print(f"Database info is a dict with {len(data)} classes:")
        print(f"Classes: {list(data.keys())}")
        print()

        for cls_name, cls_data in data.items():
            print(f"\nClass: {cls_name}")
            print(f"  Number of samples: {len(cls_data)}")

            if len(cls_data) > 0:
                first_sample = cls_data[0]
                print(f"  First sample keys: {list(first_sample.keys())}")

                for key, value in first_sample.items():
                    if isinstance(value, np.ndarray):
                        print(f"    {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: {type(value).__name__} = {value if not isinstance(value, (list, dict, np.ndarray)) else '...'}")
    else:
        print(f"Database info type: {type(data)}")

if __name__ == "__main__":
    # Inspect infos files
    inspect_infos("data/nuscenes/nuscenes_infos_train.pkl")
    inspect_infos("data/nuscenes/nuscenes_infos_val.pkl")

    # Inspect dbinfos file
    inspect_dbinfos("data/nuscenes/nuscenes_dbinfos_train.pkl")
