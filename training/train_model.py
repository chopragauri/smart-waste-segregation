"""
Step 3: Fine-tune YOLOv8 on custom dataset (tomato, keys, paper).

This script:
  1. Loads pretrained YOLOv8n weights
  2. Fine-tunes on the custom dataset
  3. Saves the best model weights
  4. Copies best weights to the main project folder
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

DATASET_YAML = Path(__file__).parent / "dataset" / "dataset.yaml"
PROJECT_DIR = Path(__file__).parent.parent  # smart-waste-segregation/
OUTPUT_NAME = "waste_custom"


def train():
    print("=" * 60)
    print("Fine-tuning YOLOv8n on custom waste dataset...")
    print("=" * 60)

    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML not found at {DATASET_YAML}")
        print("Run prepare_dataset.py first!")
        return

    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # Fine-tune on custom dataset
    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        name=OUTPUT_NAME,
        patience=10,        # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    # Find best weights
    best_weights = Path(f"runs/detect/{OUTPUT_NAME}/weights/best.pt")
    if not best_weights.exists():
        # Try alternate path
        for p in Path("runs/detect").rglob("best.pt"):
            best_weights = p
            break

    if best_weights.exists():
        dest = PROJECT_DIR / "yolov8n_custom.pt"
        shutil.copy2(str(best_weights), str(dest))
        print(f"\nBest model saved to: {dest}")
        print("Update your app to use 'yolov8n_custom.pt'!")
    else:
        print("WARNING: Could not find best.pt weights")
        print("Check runs/detect/ directory for training output")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
