#!/usr/bin/env python3
"""
Check if sweep information is properly added to Waymo dataset.
"""

import pickle
import sys
from pathlib import Path

def check_sweeps(pkl_path):
    """Check sweep information in a pickle file."""
    print(f"\n{'='*80}")
    print(f"Checking: {pkl_path}")
    print(f"{'='*80}\n")

    # Load pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    infos = data['infos']
    print(f"Total frames: {len(infos)}")

    if len(infos) == 0:
        print("❌ No frames found in dataset!")
        return False

    # Check first frame
    print(f"\n{'─'*80}")
    print("First Frame Information:")
    print(f"{'─'*80}")

    first_info = infos[0]
    print(f"Token: {first_info.get('token', 'N/A')}")
    print(f"Timestamp: {first_info.get('timestamp', 'N/A')}")
    print(f"LiDAR path: {first_info.get('lidar_path', 'N/A')}")

    # Check sweeps
    sweeps = first_info.get('sweeps', [])
    print(f"\nNumber of sweeps: {len(sweeps)}")

    if len(sweeps) == 0:
        print("⚠️  WARNING: No sweeps found in first frame!")
        print("   This might be normal if it's the first frame in a sequence.")
    else:
        print("✅ Sweeps found!")
        print(f"\nFirst sweep details:")
        first_sweep = sweeps[0]
        for key, value in first_sweep.items():
            if key in ['sensor2lidar_rotation', 'sensor2lidar_translation']:
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
            else:
                print(f"  {key}: {value}")

    # Statistics
    print(f"\n{'─'*80}")
    print("Sweep Statistics:")
    print(f"{'─'*80}")

    sweep_counts = [len(info.get('sweeps', [])) for info in infos]
    frames_with_sweeps = sum(1 for count in sweep_counts if count > 0)
    avg_sweeps = sum(sweep_counts) / len(sweep_counts) if sweep_counts else 0
    max_sweeps = max(sweep_counts) if sweep_counts else 0
    min_sweeps = min(sweep_counts) if sweep_counts else 0

    print(f"Frames with sweeps: {frames_with_sweeps}/{len(infos)} ({frames_with_sweeps/len(infos)*100:.1f}%)")
    print(f"Average sweeps per frame: {avg_sweeps:.2f}")
    print(f"Min sweeps: {min_sweeps}")
    print(f"Max sweeps: {max_sweeps}")

    # Show distribution
    print(f"\nSweep count distribution:")
    from collections import Counter
    distribution = Counter(sweep_counts)
    for count in sorted(distribution.keys()):
        bar = '█' * (distribution[count] // max(1, len(infos) // 50))
        print(f"  {count:2d} sweeps: {distribution[count]:4d} frames {bar}")

    # Check a few more frames
    print(f"\n{'─'*80}")
    print("Sample frames:")
    print(f"{'─'*80}")

    sample_indices = [0, len(infos)//2, -1] if len(infos) > 2 else [0]
    for idx in sample_indices:
        info = infos[idx]
        num_sweeps = len(info.get('sweeps', []))
        print(f"Frame {idx:3d}: {num_sweeps} sweeps | Token: {info.get('token', 'N/A')[:50]}")

    # Validation
    print(f"\n{'='*80}")
    print("Validation:")
    print(f"{'='*80}")

    issues = []

    # Check if sweeps have required fields
    for i, info in enumerate(infos):
        sweeps = info.get('sweeps', [])
        for j, sweep in enumerate(sweeps):
            required_fields = ['data_path', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation']
            missing_fields = [field for field in required_fields if field not in sweep]
            if missing_fields:
                issues.append(f"Frame {i}, Sweep {j}: Missing fields {missing_fields}")
                if len(issues) >= 5:  # Limit error messages
                    break
        if len(issues) >= 5:
            break

    if issues:
        print("❌ Found issues:")
        for issue in issues[:5]:
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more issues")
        return False
    else:
        print("✅ All sweeps have required fields!")

    # Check if sweep files exist
    print("\nChecking if sweep data files exist...")
    missing_files = []
    for i, info in enumerate(infos[:10]):  # Check first 10 frames
        for sweep in info.get('sweeps', []):
            data_path = sweep.get('data_path')
            if data_path and not Path(data_path).exists():
                missing_files.append(data_path)
                if len(missing_files) >= 3:
                    break
        if len(missing_files) >= 3:
            break

    if missing_files:
        print("⚠️  WARNING: Some sweep data files not found:")
        for path in missing_files[:3]:
            print(f"  - {path}")
        if len(missing_files) > 3:
            print(f"  ... and {len(missing_files)-3} more missing files")
    else:
        print("✅ Sweep data files exist!")

    return True


if __name__ == "__main__":
    # Check both train and val
    base_dir = Path("data/waymo/Waymo_mini")

    train_pkl = base_dir / "waymo_infos_train.pkl"
    val_pkl = base_dir / "waymo_infos_val.pkl"

    success = True

    if train_pkl.exists():
        success &= check_sweeps(train_pkl)
    else:
        print(f"❌ Training pickle not found: {train_pkl}")
        success = False

    if val_pkl.exists():
        success &= check_sweeps(val_pkl)
    else:
        print(f"❌ Validation pickle not found: {val_pkl}")
        success = False

    print(f"\n{'='*80}")
    if success:
        print("✅ Sweep data check PASSED!")
    else:
        print("❌ Sweep data check FAILED!")
    print(f"{'='*80}\n")

    sys.exit(0 if success else 1)
