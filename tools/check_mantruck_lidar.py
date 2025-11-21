import pickle
import os

with open('data/man-truckscenes/mantruck_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]

print("=== Main LiDAR path ===")
print(sample['lidar_path'])
print()

print("=== Number of sweeps ===")
print(len(sample['sweeps']))
print()

print("=== First 5 sweeps ===")
for i, sweep in enumerate(sample['sweeps'][:5]):
    path = sweep.get('data_path', 'N/A')
    print(f"Sweep {i}: {path}")
    if 'LIDAR' in path:
        # Extract which lidar sensor
        parts = path.split('/')
        for part in parts:
            if 'LIDAR' in part:
                print(f"  -> Sensor: {part}")
                break
print()

# Check if there are multiple lidar sensors in the sample
print("=== Checking for multiple LiDAR sensors ===")
lidar_sensors = set()
for sweep in sample['sweeps']:
    path = sweep.get('data_path', '')
    if 'LIDAR_LEFT' in path:
        lidar_sensors.add('LIDAR_LEFT')
    elif 'LIDAR_RIGHT' in path:
        lidar_sensors.add('LIDAR_RIGHT')
    elif 'LIDAR_REAR' in path:
        lidar_sensors.add('LIDAR_REAR')

print(f"LiDAR sensors found in sweeps: {lidar_sensors}")
