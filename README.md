# SmartBin — AI-Powered Intelligent Waste Segregation for Sustainable Cities

Real-time waste detection and classification system using YOLOv8, OpenCV, and Streamlit.

## Features

- **Real-Time Object Detection** — YOLOv8 inference at 15–30 FPS on CPU
- **Dual-Model Architecture** — Pretrained COCO model (80 classes) + Custom fine-tuned model (5 classes)
- **Two-Level Classification** — Bin Type (Wet/Dry/Metal) + Degradability (Biodegradable/Non-Biodegradable)
- **Multiple Input Sources** — Webcam, Mobile Camera (Continuity Camera), DroidCam
- **Green Credit Points** — Gamified eco-reward system with progressive levels
- **Analytics Dashboard** — Real-time pie charts, bar graphs, detection history with CSV export

## Tech Stack

- Python, YOLOv8 (Ultralytics), OpenCV, Streamlit, PyTorch, Plotly, Pandas

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Custom Model Training

```bash
cd training
python run_all.py
```

Downloads images, prepares YOLO dataset, and fine-tunes YOLOv8n on 5 custom classes: `tomato`, `keys`, `paper`, `pen`, `lip_balm`.

## Authors

- **Gauri Chopra** — [chopragauri](https://github.com/chopragauri)
- **Shivaansh Kaushik**

Department of Artificial Intelligence, Amity School of Engineering and Technology, Amity University, Noida

## License

**All Rights Reserved.**

Copyright (c) 2026 Gauri Chopra & Shivaansh Kaushik.

This software and associated documentation files (the "Software") are the exclusive intellectual property of the authors. No part of this Software may be reproduced, distributed, transmitted, displayed, published, or broadcast in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the authors.

Unauthorized copying, modification, merging, publishing, distribution, sublicensing, and/or selling of copies of this Software is strictly prohibited.

For permission requests, contact: chopra.gauri06@gmail.com
