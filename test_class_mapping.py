#!/usr/bin/env python3
"""
클래스 매핑 수정 테스트
"""

import pickle
import numpy as np

def fix_class_mapping(gt_names):
    """GT 클래스명을 올바른 순서로 매핑"""
    fixed_names = []
    for name in gt_names:
        # converter에서 잘못 매핑된 것을 수정
        if name == 'sign':  # converter에서 cyclist → sign으로 잘못 매핑
            fixed_names.append('cyclist')
        elif name == 'cyclist':  # converter에서 sign → cyclist로 잘못 매핑
            fixed_names.append('sign')
        else:
            fixed_names.append(name)  # vehicle, pedestrian은 정상
    return np.array(fixed_names)

def test_class_mapping():
    # Waymo 데이터 로드
    waymo_file = 'data/waymo/Waymo_processed/waymo_infos_val.pkl'
    with open(waymo_file, 'rb') as f:
        data = pickle.load(f)

    waymo_infos = data['infos']

    print("=== Class Mapping Fix Test ===\n")

    # 처음 5개 샘플 테스트
    for i in range(5):
        sample_info = waymo_infos[i]

        # 원본 클래스
        original_names = sample_info['gt_names']
        mask = sample_info["num_lidar_pts"] > 0
        original_names = original_names[mask]

        # 수정된 클래스
        fixed_names = fix_class_mapping(original_names)

        print(f"Sample {i}:")

        # 원본 분포
        orig_unique, orig_counts = np.unique(original_names, return_counts=True)
        orig_dist = dict(zip(orig_unique, orig_counts))
        print(f"  Original: {orig_dist}")

        # 수정된 분포
        fixed_unique, fixed_counts = np.unique(fixed_names, return_counts=True)
        fixed_dist = dict(zip(fixed_unique, fixed_counts))
        print(f"  Fixed:    {fixed_dist}")

        # 변경 사항 확인
        changes = 0
        for orig, fixed in zip(original_names, fixed_names):
            if orig != fixed:
                changes += 1

        print(f"  Changed:  {changes}/{len(original_names)} boxes")
        print()

    # 전체 통계
    all_original = []
    all_fixed = []

    for sample_info in waymo_infos[:50]:  # 첫 50개 샘플
        original_names = sample_info['gt_names']
        mask = sample_info["num_lidar_pts"] > 0
        original_names = original_names[mask]

        fixed_names = fix_class_mapping(original_names)

        all_original.extend(original_names)
        all_fixed.extend(fixed_names)

    print("=== Overall Statistics (first 50 samples) ===")

    # 원본 통계
    orig_unique, orig_counts = np.unique(all_original, return_counts=True)
    orig_total_dist = dict(zip(orig_unique, orig_counts))
    print(f"Original distribution: {orig_total_dist}")

    # 수정된 통계
    fixed_unique, fixed_counts = np.unique(all_fixed, return_counts=True)
    fixed_total_dist = dict(zip(fixed_unique, fixed_counts))
    print(f"Fixed distribution:    {fixed_total_dist}")

    # 변경 비율
    total_changes = sum(1 for orig, fixed in zip(all_original, all_fixed) if orig != fixed)
    print(f"Total changes: {total_changes}/{len(all_original)} ({100*total_changes/len(all_original):.1f}%)")

if __name__ == "__main__":
    test_class_mapping()