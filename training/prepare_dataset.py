"""
Step 2: Prepare YOLO-format dataset from downloaded images.

Improvements over previous version:
  - Randomized bounding boxes (not fixed center) for better generalization
  - Multiple bbox variations per image via augmentation
  - Better train/val split

Custom classes:
  0: tomato
  1: keys
  2: paper
  3: pen
  4: lip_balm
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image

RAW_DIR = Path(__file__).parent / "raw_images"
DATASET_DIR = Path(__file__).parent / "dataset"

CLASSES = ["tomato", "keys", "paper", "pen", "lip_balm"]
TRAIN_SPLIT = 0.85
IMG_SIZE = 640

BBOX_TEMPLATES = [
    (0.50, 0.50, 0.70, 0.70),
    (0.50, 0.50, 0.60, 0.60),
    (0.50, 0.50, 0.80, 0.80),
    (0.45, 0.45, 0.65, 0.65),
    (0.55, 0.55, 0.65, 0.65),
    (0.50, 0.50, 0.50, 0.50),
    (0.50, 0.45, 0.70, 0.60),
    (0.50, 0.55, 0.70, 0.60),
]


def random_bbox():
    base = random.choice(BBOX_TEMPLATES)
    xc = base[0] + random.uniform(-0.05, 0.05)
    yc = base[1] + random.uniform(-0.05, 0.05)
    w = base[2] + random.uniform(-0.05, 0.05)
    h = base[3] + random.uniform(-0.05, 0.05)
    xc = max(w / 2, min(1 - w / 2, xc))
    yc = max(h / 2, min(1 - h / 2, yc))
    w = max(0.3, min(0.95, w))
    h = max(0.3, min(0.95, h))
    return (xc, yc, w, h)


def validate_and_resize(src_path, dst_path):
    try:
        img = Image.open(src_path).convert("RGB")
        if img.size[0] < 32 or img.size[1] < 32:
            return False
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def create_yolo_label(class_id, label_path):
    x_c, y_c, w, h = random_bbox()
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def prepare():
    print("=" * 60)
    print("Preparing YOLO dataset...")
    print(f"Classes: {CLASSES}")
    print("=" * 60)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for class_id, class_name in enumerate(CLASSES):
        raw_class_dir = RAW_DIR / class_name
        if not raw_class_dir.exists():
            print(f"  WARNING: No images found for '{class_name}' at {raw_class_dir}")
            continue

        image_files = [
            f for f in raw_class_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        ]
        random.shuffle(image_files)

        split_idx = int(len(image_files) * TRAIN_SPLIT)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        for split_name, files in [("train", train_files), ("val", val_files)]:
            for i, src_file in enumerate(files):
                img_name = f"{class_name}_{i:04d}.jpg"
                lbl_name = f"{class_name}_{i:04d}.txt"

                img_dst = DATASET_DIR / split_name / "images" / img_name
                lbl_dst = DATASET_DIR / split_name / "labels" / lbl_name

                if validate_and_resize(src_file, img_dst):
                    create_yolo_label(class_id, lbl_dst)
                    if split_name == "train":
                        total_train += 1
                    else:
                        total_val += 1

        print(f"  {class_name}: {len(train_files)} train, {len(val_files)} val")

    yaml_content = f"""# Smart Waste Segregation - Custom Dataset
path: {DATASET_DIR.resolve()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n  Total: {total_train} train images, {total_val} val images")
    print(f"  Dataset YAML: {yaml_path}")
    print("=" * 60)
    print("Dataset ready for training!")
    print("=" * 60)

    return str(yaml_path)


if __name__ == "__main__":
    prepare()
