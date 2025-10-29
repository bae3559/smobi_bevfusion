#!/usr/bin/env python3
"""
Combine collage images from multiple directories vertically.
Each row will be from a different model/experiment.
"""

import os
import argparse
from pathlib import Path
from PIL import Image


def combine_images_vertically(image_paths, labels=None, output_path=None):
    """
    Combine multiple images vertically.

    Args:
        image_paths: List of image file paths
        labels: Optional list of labels to add to each image
        output_path: Path to save the combined image

    Returns:
        PIL.Image: Combined image
    """
    images = [Image.open(p) for p in image_paths]

    # Get dimensions
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    # Use the maximum width and sum of heights
    max_width = max(widths)
    total_height = sum(heights)

    # Add space for labels if provided
    label_height = 40 if labels else 0
    total_height += label_height * len(images)

    # Create new image
    combined = Image.new('RGB', (max_width, total_height), color='white')

    # Paste images
    y_offset = 0
    for i, img in enumerate(images):
        # Add label if provided
        if labels:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            except:
                font = ImageFont.load_default()

            # Draw label background
            draw.rectangle([(0, y_offset), (max_width, y_offset + label_height)], fill='black')
            # Draw label text
            draw.text((10, y_offset + 5), labels[i], fill='white', font=font)
            y_offset += label_height

        # Center image horizontally if needed
        x_offset = (max_width - img.width) // 2
        combined.paste(img, (x_offset, y_offset))
        y_offset += img.height

    if output_path:
        combined.save(output_path)

    return combined


def main():
    parser = argparse.ArgumentParser(description='Combine collage images vertically')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='List of directories containing collage images')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each directory (optional)')
    parser.add_argument('--output-dir', type=str, default='visualize/combined_collages',
                        help='Output directory for combined images')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to process')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files from first directory
    first_dir = Path(args.dirs[0])
    image_files = sorted([f.name for f in first_dir.glob('*.png')])

    if args.limit:
        image_files = image_files[:args.limit]

    print(f"Found {len(image_files)} images to combine")
    print(f"Combining from {len(args.dirs)} directories:")
    for i, d in enumerate(args.dirs):
        label = args.labels[i] if args.labels else f"Dir {i+1}"
        print(f"  {label}: {d}")

    # Process each image
    for i, img_name in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_name}")
        image_paths = []

        # Collect paths from all directories
        skip = False
        for dir_path in args.dirs:
            img_path = Path(dir_path) / img_name
            if not img_path.exists():
                print(f"Warning: {img_path} not found, skipping this frame")
                skip = True
                break
            image_paths.append(img_path)

        if skip:
            continue

        # Combine images
        output_path = output_dir / img_name
        combine_images_vertically(
            image_paths,
            labels=args.labels,
            output_path=output_path
        )

    print(f"\nDone! Combined images saved to: {output_dir}")


if __name__ == '__main__':
    main()
