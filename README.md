<div align="center">

# ♻️ SmartBin — AI-Powered Waste Segregation

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Real-time waste detection and classification system**  
Dual-model YOLOv8 architecture with gamified eco-rewards and live analytics dashboard.

</div>

---

## 📌 About

SmartBin uses computer vision to detect and classify waste in real-time through a webcam or mobile camera. It runs two models simultaneously — a pretrained COCO model for broad object detection and a custom fine-tuned model for specific waste categories — then maps each detected object to a bin type and degradability label.

---

## ✨ Features

- **Real-Time Detection** — YOLOv8 inference at 15–30 FPS on CPU
- **Dual-Model Architecture** — Pretrained COCO (80 classes) + Custom fine-tuned model (5 classes) running in parallel
- **Two-Level Classification** — Bin Type (Wet / Dry / Metal) + Degradability (Biodegradable / Non-Biodegradable)
- **Multiple Input Sources** — Webcam, Mobile Camera (Continuity Camera), DroidCam
- **Green Credit Points** — Gamified eco-reward system with progressive levels to encourage correct waste disposal
- **Analytics Dashboard** — Live pie charts, bar graphs, detection history with CSV export

---

## 🗂️ Project Structure

```
smart-waste-segregation/
├── app.py                  # Streamlit web app — camera feed, UI, dashboard
├── waste_classifier.py     # Dual-model inference + classification logic
├── yolov8n_custom.pt       # Custom fine-tuned YOLOv8 weights
├── requirements.txt
├── runtime.txt
├── setup.sh
└── training/
    ├── download_images.py  # Scrape training images for 5 custom classes
    ├── prepare_dataset.py  # Build YOLO-format dataset with train/val split
    ├── train_model.py      # Fine-tune YOLOv8n on custom dataset
    └── run_all.py          # One-shot script: download → prepare → train
```

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| **Detection** | YOLOv8 (Ultralytics), OpenCV |
| **Deep Learning** | PyTorch |
| **UI & Dashboard** | Streamlit, Plotly |
| **Data Handling** | Pandas |
| **Language** | Python 3.9+ |

---

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/chopragauri/smart-waste-segregation.git
cd smart-waste-segregation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🏋️ Custom Model Training

The custom model is fine-tuned on 5 waste classes: `tomato`, `keys`, `paper`, `pen`, `lip_balm`

```bash
cd training
python run_all.py
```

This runs the full pipeline: downloads training images → builds YOLO dataset → fine-tunes YOLOv8n. The trained weights are saved and loaded by `waste_classifier.py`.

---

## 👩‍💻 Built By

**Gauri Chopra** — [chopragauri](https://github.com/chopragauri)  
**Shivaansh Kaushik**  

Department of Artificial Intelligence, Amity University, Noida

---

## 📄 License

**All Rights Reserved.**

Copyright (c) 2026 Gauri Chopra & Shivaansh Kaushik.

This software and associated documentation files (the "Software") are the exclusive intellectual property of the authors. No part of this Software may be reproduced, distributed, transmitted, displayed, published, or broadcast in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the authors.

Unauthorized copying, modification, merging, publishing, distribution, sublicensing, and/or selling of copies of the Software is strictly prohibited.

For permission requests, contact: chopra.gauri06@gmail.com
