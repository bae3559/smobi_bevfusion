import pickle
import numpy as np

# Load mantruck data
with open('data/man-truckscenes/mantruck_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)

# Check first sample
sample = data['infos'][0]
print('=== Sample keys ===')
print(list(sample.keys()))
print()

print('=== GT boxes ===')
print('gt_boxes shape:', sample['gt_boxes'].shape)
print('First 2 boxes:')
for i in range(min(2, len(sample['gt_boxes']))):
    box = sample['gt_boxes'][i]
    print(f'  Box {i}: x={box[0]:.2f}, y={box[1]:.2f}, z={box[2]:.2f}, dx={box[3]:.2f}, dy={box[4]:.2f}, dz={box[5]:.2f}, yaw={box[6]:.2f}')
print()

print('=== Camera info ===')
cam_keys = list(sample['cams'].keys())
print('Camera names:', cam_keys)
print()

# Check first camera
first_cam_key = cam_keys[0]
first_cam = sample['cams'][first_cam_key]
print(f'=== First camera: {first_cam_key} ===')
print('Keys:', list(first_cam.keys()))
print()

if 'sensor2lidar_rotation' in first_cam:
    rot = np.array(first_cam['sensor2lidar_rotation'])
    print('sensor2lidar_rotation shape:', rot.shape)
    print('sensor2lidar_rotation:')
    print(rot)
    print()

if 'sensor2lidar_translation' in first_cam:
    trans = np.array(first_cam['sensor2lidar_translation'])
    print('sensor2lidar_translation:', trans)
    print()

if 'cam_intrinsic' in first_cam:
    intrinsic = np.array(first_cam['cam_intrinsic'])
    print('cam_intrinsic shape:', intrinsic.shape)
    print('cam_intrinsic:')
    print(intrinsic)
    print()

print('=== Metadata ===')
if 'metadata' in data:
    print('Metadata:', data['metadata'])
