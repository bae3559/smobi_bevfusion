import struct
import numpy as np

# Parse PCD header and extract a few points manually
pcd_path = 'data/man-truckscenes/samples/LIDAR_TOP_FRONT/LIDAR_TOP_FRONT_1692868171700983.pcd'

with open(pcd_path, 'rb') as f:
    # Read header
    header_lines = []
    while True:
        line = f.readline().decode('ascii').strip()
        header_lines.append(line)
        if line.startswith('DATA'):
            break

    print("=== Header ===")
    for line in header_lines:
        print(line)

    # After DATA binary_compressed, the rest is compressed data
    # For binary_compressed format, we can't easily parse it manually
    # But we already know from the header: x y z intensity timestamp

print("\n=== Analysis ===")
print("Fields: x, y, z, intensity, timestamp")
print("All float32 except timestamp which is uint64")
print("From earlier debug output, we know x,y,z are in meter range (sensor frame)")
