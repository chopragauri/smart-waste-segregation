"""
Step 1: Download training images for custom classes.
Uses bing_image_downloader to fetch images from the web.

Custom classes: tomato, keys, paper, pen, lip_balm
"""

import os
import shutil
from pathlib import Path
from bing_image_downloader import downloader

BASE_DIR = Path(__file__).parent / "raw_images"

QUERIES = {
    "tomato": [
        "single tomato fruit",
        "tomato on table",
        "red tomato close up",
        "tomato vegetable photo",
        "fresh tomato",
        "tomato on cutting board",
        "cherry tomato close up",
        "tomato kitchen counter",
    ],
    "keys": [
        "metal key single",
        "house key on table",
        "keys bunch photo",
        "metal keys close up",
        "door key photo",
        "car keys on table",
        "key chain photo",
        "single brass key",
    ],
    "paper": [
        "sheet of paper on desk",
        "crumpled paper waste",
        "paper sheet white",
        "notebook paper photo",
        "paper on table",
        "stack of paper sheets",
        "paper document on desk",
        "loose paper pages",
    ],
    "pen": [
        "ballpoint pen on desk",
        "single pen photo",
        "pen on paper close up",
        "plastic pen photo",
        "writing pen on table",
        "blue pen close up",
        "pen isolated white background",
        "ball pen photo",
    ],
    "lip_balm": [
        "lip balm tube photo",
        "chapstick on table",
        "lip balm close up",
        "small lip balm product",
        "lip balm stick photo",
        "chapstick tube close up",
        "lip balm product isolated",
        "lip balm plastic tube",
    ],
}

IMAGES_PER_QUERY = 30


def download_all():
    print("=" * 60)
    print("Downloading training images...")
    print("=" * 60)

    for class_name, queries in QUERIES.items():
        class_dir = BASE_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Downloading images for: {class_name} ---")

        for i, query in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}: '{query}'")
            try:
                downloader.download(
                    query,
                    limit=IMAGES_PER_QUERY,
                    output_dir=str(BASE_DIR / "_temp"),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=10,
                )
                temp_dir = BASE_DIR / "_temp" / query
                if temp_dir.exists():
                    for j, img_file in enumerate(temp_dir.iterdir()):
                        if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                            dest = class_dir / f"{class_name}_{i}_{j}{img_file.suffix}"
                            shutil.move(str(img_file), str(dest))
            except Exception as e:
                print(f"    Warning: {e}")

        count = len([f for f in class_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')])
        print(f"  Total images for {class_name}: {count}")

    temp_dir = BASE_DIR / "_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("Download complete! Images saved to:", BASE_DIR)
    print("=" * 60)


if __name__ == "__main__":
    download_all()
