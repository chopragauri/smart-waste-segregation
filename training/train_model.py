"""
Step 3: Fine-tune YOLOv8 on custom dataset (tomato, keys, paper, pen, lip_balm).

Trains with aggressive augmentation for better generalization from auto-labeled data.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = Path(__file__).parent / "dataset" / "dataset.yaml"
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_NAME = "waste_custom"


def train():
    print("=" * 60)
    print("Fine-tuning YOLOv8n on custom waste dataset...")
    print("=" * 60)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML not found at {DATASET_YAML}")
        print("Run prepare_dataset.py first!")
        return

    weights = str(Path(__file__).parent / "yolov8n.pt")
    model = YOLO(weights)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        name=OUTPUT_NAME,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        # Aggressive augmentation to compensate for auto-labels
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
    )

    best_weights = Path(f"runs/detect/{OUTPUT_NAME}/weights/best.pt")
    if not best_weights.exists():
        for p in Path("runs/detect").rglob("best.pt"):
            best_weights = p
            break

    if best_weights.exists():
        dest = PROJECT_DIR / "yolov8n_custom.pt"
        shutil.copy2(str(best_weights), str(dest))
        print(f"\nBest model saved to: {dest}")
    else:
        print("WARNING: Could not find best.pt weights")
        print("Check runs/detect/ directory for training output")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
