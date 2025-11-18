#!/usr/bin/env python3
"""Simple check for sweep data in pickle files (no dependencies)"""

import pickle
import sys

def check_sweeps(pkl_path, dataset_name):
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"{'='*60}")

    try:
        # Unpickle with encoding for compatibility
        with open(pkl_path, 'rb') as f:
            # Try to load without numpy first
            import pickletools
            data = pickle.load(f)

        print(f"✓ Loaded: {pkl_path}")

        if 'infos' in data:
            infos = data['infos']
            print(f"Total samples: {len(infos)}")

            # Check first 5 samples
            for i in range(min(5, len(infos))):
                info = infos[i]
                print(f"\nSample {i}:")
                print(f"  Keys: {list(info.keys())}")

                if 'sweeps' in info:
                    sweeps = info['sweeps']
                    print(f"  Sweeps: {len(sweeps)} frames")

                    if len(sweeps) > 0:
                        # Print first sweep structure
                        first_sweep = sweeps[0]
                        print(f"  First sweep keys: {list(first_sweep.keys())}")
                        if 'data_path' in first_sweep:
                            print(f"  First sweep path: {first_sweep['data_path']}")
                else:
                    print(f"  ✗ NO 'sweeps' key!")

        else:
            print("✗ No 'infos' key in pickle file")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_sweeps("data/nuscenes/nuscenes_infos_val.pkl", "NuScenes Validation")
    check_sweeps("data/waymo/Waymo_mini/waymo_infos_val.pkl", "Waymo Validation")
