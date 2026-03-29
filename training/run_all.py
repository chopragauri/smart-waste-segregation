"""
Master script: Download images → Prepare dataset → Train YOLOv8.
Run this once to create the custom fine-tuned model.
"""

from download_images import download_all
from prepare_dataset import prepare
from train_model import train


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STEP 1/3: Downloading images")
    print("=" * 60)
    download_all()

    print("\n" + "=" * 60)
    print("  STEP 2/3: Preparing YOLO dataset")
    print("=" * 60)
    prepare()

    print("\n" + "=" * 60)
    print("  STEP 3/3: Training YOLOv8")
    print("=" * 60)
    train()

    print("\n\nALL DONE! Your custom model is at: yolov8n_custom.pt")
