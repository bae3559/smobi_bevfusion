#!/usr/bin/env python3
"""Check NuScenes sweep data in info files"""

import pickle
import sys

def check_sweeps(pkl_path):
    print(f"Loading: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Keys: {data.keys()}")

    if 'infos' in data:
        infos = data['infos']
        print(f"Total samples: {len(infos)}")

        # Check first 10 samples
        for i in range(min(10, len(infos))):
            info = infos[i]

            if 'sweeps' in info:
                num_sweeps = len(info['sweeps'])
                print(f"Sample {i}: {num_sweeps} sweeps")

                if i == 0 and num_sweeps > 0:
                    # Print first sweep details
                    print(f"  First sweep keys: {info['sweeps'][0].keys()}")
                    print(f"  First sweep data_path: {info['sweeps'][0].get('data_path', 'N/A')}")
            else:
                print(f"Sample {i}: NO 'sweeps' key!")

        # Count sweep distribution
        sweep_counts = {}
        for info in infos:
            count = len(info.get('sweeps', []))
            sweep_counts[count] = sweep_counts.get(count, 0) + 1

        print(f"\nSweep distribution:")
        for count in sorted(sweep_counts.keys()):
            print(f"  {count} sweeps: {sweep_counts[count]} samples")
    else:
        print("No 'infos' key in pickle file!")

if __name__ == "__main__":
    print("=" * 60)
    print("NuScenes Validation")
    print("=" * 60)
    check_sweeps("data/nuscenes/nuscenes_infos_val.pkl")

    print("\n" + "=" * 60)
    print("Waymo Validation")
    print("=" * 60)
    check_sweeps("data/waymo/Waymo_mini/waymo_infos_val.pkl")
