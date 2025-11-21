from truckscenes.truckscenes import TruckScenes

nusc = TruckScenes(version='v1.0-mini', dataroot='data/man-truckscenes/', verbose=True)

# Get first sample
sample = nusc.sample[0]

print("=== Available sensors in sample ===")
for sensor_name in sorted(sample['data'].keys()):
    print(f"  - {sensor_name}")

print("\n=== LIDAR sensors ===")
lidar_sensors = [s for s in sample['data'].keys() if 'LIDAR' in s]
for sensor in sorted(lidar_sensors):
    print(f"  - {sensor}")

print(f"\nTotal LIDAR sensors: {len(lidar_sensors)}")
