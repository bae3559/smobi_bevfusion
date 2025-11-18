#!/usr/bin/env python3
"""Debug sweep generation."""

import pickle
from pathlib import Path

pkl_path = "data/waymo/Waymo_mini/waymo_infos_val.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

infos = data['infos']

print(f"Total frames: {len(infos)}")
print()

# Check frame 10 (should have ~9 sweeps)
if len(infos) > 10:
    frame_10 = infos[10]
    print(f"Frame 10:")
    print(f"  Token: {frame_10['token']}")
    print(f"  Sweeps: {len(frame_10['sweeps'])}")
    print()

    # Check sweep timestamps
    frame_ts = frame_10['timestamp']
    print(f"  Frame timestamp: {frame_ts}")
    print(f"  Sweep timestamps:")
    for i, sweep in enumerate(frame_10['sweeps']):
        sweep_ts = sweep['timestamp']
        time_diff = (frame_ts - sweep_ts) / 1e6  # Convert to seconds
        print(f"    Sweep {i}: {sweep_ts} (Δt = {time_diff:.3f}s)")
    print()

# Check all frames for sequence changes
print("Checking sequences:")
sequences = set()
for i, info in enumerate(infos):
    token = info['token']
    seq_id = token.split('_')[0]  # Extract sequence ID
    sequences.add(seq_id)

print(f"Number of sequences: {len(sequences)}")
print(f"Sequence IDs: {sequences}")
print()

# Check where sequences change
print("Sequence changes:")
prev_seq = None
for i, info in enumerate(infos):
    token = info['token']
    seq_id = token.split('_')[0]

    if seq_id != prev_seq:
        num_sweeps = len(info['sweeps'])
        print(f"  Frame {i}: New sequence {seq_id} (sweeps: {num_sweeps})")
        prev_seq = seq_id
