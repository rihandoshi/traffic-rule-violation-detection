# Traffic Rule Violation Detection — Person 1 Guide (Core Backbone)

This document is your **implementation playbook** for the role described as *Person 1*: repository structure, two-wheeler + person detection, rider–bike association, shared data formats, and utilities. It is written so you can **learn by building** without copy-pasting a full solution.

**Course constraints (keep in mind while you design):**

- All model files under `models/` together must be **≤ 250 MB**.
- **No huge VLM** (e.g. >1B parameters).
- **Offline** at evaluation — no downloading weights at runtime.
- `TrafficViolationDetector.__init__` loads everything; `predict()` must be **stateless**.

---

## Table of contents

1. [What you own vs. what teammates own](#1-what-you-own-vs-what-teammates-own)
2. [Recommended mental model of the pipeline](#2-recommended-mental-model-of-the-pipeline)
3. [Day-zero setup (environment)](#3-day-zero-setup-environment)
4. [Repository layout to create](#4-repository-layout-to-create)
5. [Implementation order (today EOD checklist)](#5-implementation-order-today-eod-checklist)
6. [Detection: bikes and riders](#6-detection-bikes-and-riders)
7. [Association: riders ↔ bikes (the hard part)](#7-association-riders--bikes-the-hard-part)
8. [Interfaces for Person 2 and Person 3](#8-interfaces-for-person-2-and-person-3)
9. [Utilities you should implement](#9-utilities-you-should-implement)
10. [Testing without ground truth](#10-testing-without-ground-truth)
11. [Glossary (terms people throw around)](#11-glossary-terms-people-throw-around)
12. [Learning resources](#12-learning-resources)
13. [Common failure modes](#13-common-failure-modes)
14. [Function templates (signatures only)](#14-function-templates-signatures-only)

---

## 1. What you own vs. what teammates own

| You (Person 1) | Person 2 (helmet) | Person 3 (plate + OCR + final JSON) |
|----------------|-------------------|--------------------------------------|
| Repo skeleton, `solution.py` class shell | Helmet / no-helmet model | Plate localization + OCR |
| Bike detection | Per-rider helmet inference | Combining counts + plate into spec JSON |
| Person/rider detection | `helmet_violations` per bike | `requirements.txt` / packaging polish |
| Rider–bike grouping | Robustness on hard heads | Pre/post-processing for OCR |
| Cropping + geometry utils | | |

**Your job today:** everything through **“for each bike, here are the rider crops (and bike crop) in a clean structure.”** Helmet and plate strings can be placeholders until they merge.

---

## 2. Recommended mental model of the pipeline

```text
RGB image
   → resize / letterbox (optional, for detector)
   → bike detector  → list of bike boxes
   → person detector → list of person boxes
   → associate persons to bikes
   → for each bike: package crops + metadata for downstream modules
```

Downstream modules should not need to re-run detection if you give them **consistent crops and coordinates**.

---

## 3. Day-zero setup (environment)

**Suggested stack (fits course constraints and team plan):**

- **Python 3.10+** (3.10 or 3.11 are widely supported by CV libs).
- **Ultralytics YOLOv8** (`ultralytics`) for detection — fast to prototype, good export options, small models (`yolov8n`, `yolov8s`).
- **OpenCV** (`opencv-python`) for I/O, drawing, cropping, basic transforms.
- **NumPy** for arrays and box math.

**Why YOLO here:** single forward pass, good speed/size tradeoff, COCO includes `person`, and you can use a **motorcycle/scooter** class from COCO or a **custom fine-tuned** head if your course allows training (optional for day 1: start with COCO `motorcycle` + heuristic for scooters, or a small custom dataset later).

**Install pattern (you run this locally):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install ultralytics opencv-python numpy
```

Pin versions in `requirements.txt` once things work (Person 3 often owns the final file, but you can start a minimal one).

---

## 4. Repository layout to create

Create something like:

```text
project/
├── solution.py              # TrafficViolationDetector + thin orchestration
├── requirements.txt
├── README.md
├── models/                  # weights only; total size ≤ 250 MB
│   └── .gitkeep             # optional: keep folder in git if weights are gitignored
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── bikes.py       # load bike model, run inference
│   │   └── riders.py      # load person model, run inference
│   ├── association/
│   │   ├── __init__.py
│   │   └── assign.py      # riders per bike logic
│   ├── helmet/            # empty or stub until Person 2 merges
│   ├── ocr/               # empty or stub until Person 3 merges
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       └── geometry.py
├── outputs/                 # debug visualizations (gitignored)
└── test_images/             # a few local images for sanity checks
```

**Tip:** Keep `solution.py` thin: construct detector paths, call your modules, return the dict shape the course requires (even if helmet/OCR are stubs at first).

---

## 5. Implementation order (today EOD checklist)

Work in this order so you are never blocked:

1. **Utils:** load image, clamp boxes to image bounds, crop, optional draw.
2. **Bike detector wrapper:** `image → List[BBox]` with scores/classes if needed.
3. **Person detector wrapper:** same.
4. **Association:** `bikes + persons → grouped structure`.
5. **Wire `TrafficViolationDetector.predict`:** run steps 2–4; for each violating group (or each bike), build the JSON entries (initially `helmet_violations` can be `0` and `license_plate` `""` until teammates plug in).
6. **Visual debugging:** save images with bike boxes, person boxes, and IDs/colors per group under `outputs/`.

If EOD hits and association is shaky, **ship a deterministic baseline** (documented in README) rather than crashing — the spec says runtime errors zero the sample.

---

## 6. Detection: bikes and riders

### 6.1 Bounding box format (pick one and stick to it)

Common choices:

- **XYXY:** `[x1, y1, x2, y2]` pixel coords, origin top-left.
- **XYWH:** `[x, y, w, h]`.

**Recommendation:** use **XYXY float** internally; convert at API boundaries if needed.

### 6.2 Two-wheeler class strategy

**Option A — COCO pretrained only (fastest):**

- COCO has `motorcycle`. Scooters may be missed or labeled oddly; you may need heuristics or augmentation later.

**Option B — Fine-tuned detector (stronger, more work):**

- Collect images, label `motorcycle`, `scooter` (or single `two_wheeler` class).
- Train YOLOv8n; keep an eye on **`.pt` file size** + any exported ONNX.

**Practical path for today:** get **Option A** running end-to-end; note failure cases for the report.

### 6.3 Person detection

COCO `person` is standard. Use NMS (usually inside YOLO) to reduce duplicate boxes.

### 6.4 Multi-model size budget

Two small YOLO weights (e.g. nano) + teammates’ models must stay under **250 MB total**. Track sizes:

```bash
# example (Windows PowerShell): sum file sizes in models/
Get-ChildItem models -Recurse -File | Measure-Object -Property Length -Sum
```

---

## 7. Association: riders ↔ bikes (the hard part)

### 7.1 What “association” means

You must decide: **this person box “rides” this motorcycle box**, possibly with **multiple riders** on one bike.

There is **no universal perfect rule** without tracking or 3D reasoning; you want **simple geometry** that works on most street scenes.

### 7.2 Features you can use (2D image)

| Idea | Intuition | Pros | Cons |
|------|-----------|------|------|
| **IoU overlap** | Person overlaps bike bbox | Very simple | Legs/arms stick out; overlap can be small |
| **Center inside** | Person center point falls inside expanded bike box | Stable | Misses if bbox is tight |
| **Bottom-center / foot point** | Project “feet” near bottom of person box into bike region | Often better for “sitting on” | Needs tuning |
| **Distance (centers)** | Assign nearest bike within radius | Handles overlap issues | Can wrong-assign in crowds |
| **Combination score** | Weighted mix of IoU + distance + vertical alignment | More robust | More knobs |

**Suggested baseline algorithm (you implement and tune constants):**

1. Expand bike bbox slightly (e.g. 10–25% margin) — **expanded bike region**.
2. For each person, compute a **contact point**: e.g. bottom-center `( (x1+x2)/2, y2 )`.
3. If contact point is inside expanded bike region → candidate for that bike.
4. If a person matches **multiple** bikes, assign to **highest IoU** or **smallest center distance**.
5. If a person matches **no** bike, leave **unassigned** (do not invent riders).

**Edge cases to think about:**

- **Side-by-side bikes:** distance + IoU together helps.
- **Occlusion:** person box may be truncated; expansion helps.
- **Person walking near bike:** foot point might still fall inside expanded box — use **minimum IoU threshold** or **require overlap > small epsilon**.

### 7.3 Counting riders per bike

After assignment:

- `num_riders = len(assigned_persons_for_this_bike)`.

**Violations (for final JSON, once helmet is merged):**

- `num_riders > 2` → violation type in problem statement.
- `helmet_violations > 0` → violation.
- Combined case is implied by those counts.

**Important:** The spec’s output lists **violations** — clarify with your team whether you output **only violating bikes** or **all bikes**. The field name `violations` strongly suggests **only cases that violate a rule**; if in doubt, match whatever the teaching team’s reference implementation expects (ask on forum if ambiguous).

---

## 8. Interfaces for Person 2 and Person 3

Define **plain dicts / dataclasses** everyone imports. Example **conceptual** shapes (adapt names to your code):

**Per rider (after association):**

```python
# Conceptual — not a full implementation
rider_record = {
    "bbox_xyxy": [float, float, float, float],
    "crop_bgr": None,  # np.ndarray (H,W,3) or lazy-cropped later
    "score": float,
}
```

**Per bike group:**

```python
bike_group = {
    "bike_bbox_xyxy": [...],
    "bike_score": float,
    "riders": [rider_record, ...],
}
```

**Helmet module (Person 2) — expected call pattern:**

```python
# helmet_violations = count riders where helmet_score < threshold
infer_helmet_on_rider_crop(rider_crop_bgr) -> bool
```

**OCR module (Person 3) — expected call pattern:**

```python
# Often: detect plate inside bike crop OR full image, then OCR
extract_plate_text(bike_crop_bgr_or_full_image) -> str
```

**Your responsibility:** produce **good crops** (optional: also pass **full image** + coords so OCR can re-crop with context).

---

## 9. Utilities you should implement

Minimum useful helpers:

- `load_image_bgr(path) -> np.ndarray`
- `clamp_bbox_xyxy(box, width, height) -> box`
- `crop_xyxy(image, box) -> np.ndarray`
- `iou_xyxy(a, b) -> float`
- `expand_bbox_xyxy(box, image_wh, scale=0.2) -> box` (margin)
- `draw_boxes(image, boxes, labels=...) -> np.ndarray` (debug)

Optional but nice:

- Letterbox / undo-letterbox if you need to map coords between resized and original image.

---

## 10. Testing without ground truth

**Sanity tests you can do today:**

- Run on 5–10 downloaded street images; visualize assignments.
- **Invariant checks:** assigned rider indices unique across bikes (unless you explicitly allow sharing — usually **no**).
- **Crash tests:** empty detections, huge image, tiny image, grayscale (convert to BGR).

**Synthetic toy test (optional):**

- Paste simple rectangles in a blank image to verify IoU / center rules behave as you expect.

---

## 11. Glossary (terms people throw around)

| Term | Meaning |
|------|---------|
| **BBox** | Bounding box rectangle around an object. |
| **XYXY / XYWH** | Two common bbox parameterizations; convert carefully. |
| **Confidence score** | Model’s estimated probability for a detection (0–1). |
| **NMS (Non-Maximum Suppression)** | Removes duplicate boxes for the same object. |
| **IoU (Intersection over Union)** | Overlap metric between two boxes; 0 none, 1 identical. |
| **mAP** | Mean Average Precision; common detection benchmark (you may not need it for the course). |
| **Letterboxing** | Resize image to fit model input while preserving aspect ratio, padding bars. |
| **COCO** | Common dataset with 80 classes including `person`, `motorcycle`. |
| **Fine-tuning** | Start from pretrained weights and train a bit on your data. |
| **ONNX / TensorRT / OpenVINO** | Export formats for faster inference; watch compatibility offline. |
| **Stateless `predict`** | No reliance on previous images; no hidden counters unless reset each call. |

---

## 12. Learning resources

**YOLOv8 / Ultralytics**

- Official docs: [https://docs.ultralytics.com](https://docs.ultralytics.com) — start with Predict, Models, Python usage.

**OpenCV**

- `cv2.imread`, `cv2.rectangle`, `cv2.resize`, color space notes (BGR vs RGB).

**Bounding box math**

- Any tutorial on IoU; implement yourself once, reuse forever.

**Association / heuristics**

- Search: “bbox association heuristic pedestrian vehicle” — read for ideas, but keep your implementation **simple and documented**.

**Robust engineering**

- Think about: timeouts, empty lists, `predict` always returning valid JSON keys.

---

## 13. Common failure modes

1. **Coordinate mismatch:** resized image for model but forgot to map boxes back to original — crops are nonsense.
2. **Wrong violation filtering:** output structure does not match autograder expectations.
3. **Model path bugs:** works on your PC, fails offline on evaluator if paths are wrong — use `model_dir` argument.
4. **OOM / slow:** huge images; consider downscaling for detection stage.
5. **Over-merging riders:** one bike “steals” all people — fix with distance gates and IoU thresholds.

---

## 14. Function templates (signatures only)

Use these as **skeletons**; fill logic yourself.

```python
# src/utils/image.py
from typing import Tuple
import numpy as np

def load_image_bgr(path: str) -> np.ndarray:
    ...

def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    ...
```

```python
# src/utils/geometry.py
from typing import List, Tuple
import numpy as np

BBox = Tuple[float, float, float, float]

def clamp_bbox_xyxy(box: BBox, width: int, height: int) -> BBox:
    ...

def crop_xyxy(image: np.ndarray, box: BBox) -> np.ndarray:
    ...

def iou_xyxy(a: BBox, b: BBox) -> float:
    ...

def expand_bbox_xyxy(box: BBox, width: int, height: int, margin_ratio: float) -> BBox:
    ...
```

```python
# src/detection/bikes.py
from typing import List, Tuple
import numpy as np

BBox = Tuple[float, float, float, float]

class BikeDetector:
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        ...

    def predict(self, image_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        """Return list of (box, score, class_name)."""
        ...
```

```python
# src/detection/riders.py
from typing import List, Tuple
import numpy as np

BBox = Tuple[float, float, float, float]

class RiderDetector:
    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        ...

    def predict(self, image_bgr: np.ndarray) -> List[Tuple[BBox, float]]:
        """Return list of (box, score) for persons."""
        ...
```

```python
# src/association/assign.py
from typing import List, Dict, Any, Tuple

BBox = Tuple[float, float, float, float]

def associate_riders_to_bikes(
    bike_boxes: List[Tuple[BBox, float]],
    rider_boxes: List[Tuple[BBox, float]],
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of groups:
    {
      "bike_bbox": [x1,y1,x2,y2],
      "bike_score": float,
      "riders": [{"bbox": [...], "score": float}, ...],
    }
    """
    ...
```

```python
# solution.py (shell only)
class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models") -> None:
        ...

    def predict(self, image_path: str) -> dict:
        """
        Return exactly:
        {
          "violations": [
            {
              "num_riders": int,
              "helmet_violations": int,
              "license_plate": str,
            },
            ...
          ]
        }
        """
        ...
```

---

## Final notes for you

- **Document your association rules** (constants, thresholds) in a short “Design decisions” section your team can cite in the report.
- **Prioritize no crashes + clear interfaces** over perfect accuracy on day one.
- When Person 2 and 3 land, they should only touch `src/helmet/`, `src/ocr/`, and integration inside `solution.py` — that’s the payoff for keeping your backbone clean.

Good luck — you’re building the part everyone else plugs into; if this layer is stable, the project moves fast.
