"""
Step 1: Download training images for custom classes (tomato, keys, paper).
Uses bing_image_downloader to fetch images from the web.
"""

import os
import shutil
from pathlib import Path
from bing_image_downloader import downloader

BASE_DIR = Path(__file__).parent / "raw_images"

# Search queries per class — multiple queries for diversity
QUERIES = {
    "tomato": [
        "single tomato fruit",
        "tomato on table",
        "red tomato close up",
        "tomato vegetable photo",
        "fresh tomato",
    ],
    "keys": [
        "metal key single",
        "house key on table",
        "keys bunch photo",
        "metal keys close up",
        "door key photo",
    ],
    "paper": [
        "sheet of paper on desk",
        "crumpled paper waste",
        "paper sheet white",
        "notebook paper photo",
        "paper on table",
    ],
}

IMAGES_PER_QUERY = 25  # ~25 per query x 5 queries = ~125 per class


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
                # Move downloaded images into class folder
                temp_dir = BASE_DIR / "_temp" / query
                if temp_dir.exists():
                    for j, img_file in enumerate(temp_dir.iterdir()):
                        if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                            dest = class_dir / f"{class_name}_{i}_{j}{img_file.suffix}"
                            shutil.move(str(img_file), str(dest))
            except Exception as e:
                print(f"    Warning: {e}")

        # Count images
        count = len(list(class_dir.glob("*")))
        print(f"  Total images for {class_name}: {count}")

    # Cleanup temp
    temp_dir = BASE_DIR / "_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("Download complete! Images saved to:", BASE_DIR)
    print("=" * 60)


if __name__ == "__main__":
    download_all()
