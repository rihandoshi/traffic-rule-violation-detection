# Traffic Rule Violation Detection 🏍️🚦

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/rihandoshi/traffic-rule-violation-detection)

A modular, robust, and highly efficient computer vision system designed to detect traffic rule violations involving two-wheelers from street images. This project was developed to satisfy the constraints of the AID 728 Course Project.

> 📝 **For an in-depth explanation of the methodology, architecture, and mathematical constraints, please view the [cv_report.pdf](./cv_report.pdf) file.**

## 🌟 Features
- **Vehicle & Rider Detection:** Utilizes dynamic resolution YOLOv8 to accurately detect motorcycles, scooters, and riders.
- **3D Depth-Aware Association:** Integrates `Depth-Anything-V2-Small` to filter out background pedestrians, logically mapping riders to bikes using true 3D spatial reasoning.
- **Helmet Detection:** Features a custom fine-tuned YOLO model (`helmet_yolov8s.pt`) with dynamic confidence thresholding for distant crops.
- **License Plate OCR:** Extracts license plates of violating vehicles using PaddleOCR backed by extensive morphological noise reduction and deskewing.
- **Offline & Lightweight:** Fully offline execution with model weights heavily optimized (< 150MB total) to fit strict academic constraints.

## 🛠️ Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/rihandoshi/traffic-rule-violation-detection.git
cd traffic-rule-violation-detection
pip install -r requirements.txt
```

*Note: You must ensure all required `.pt` weights are located inside the `models/` directory before execution.*

## 🚀 Usage

The main interface is the `TrafficViolationDetector` class. 

```python
from solution import TrafficViolationDetector

# Initialize the detector (loads models offline)
detector = TrafficViolationDetector(model_dir="./models")

# Run inference on an image
result = detector.predict("test_images/1.jpg")
print(result)
```

## 🧠 Fine-Tuning
A dedicated training script `train_helmet.py` is included in the root directory. This demonstrates the exact reproducible pipeline used to fine-tune the base YOLO architecture on our custom motorcycle helmet datasets over 25 epochs.
