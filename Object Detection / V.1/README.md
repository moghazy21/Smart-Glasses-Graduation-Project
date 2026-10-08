# Object Detection & Distance Alerts

Real-time obstacle and person detection with spoken proximity warnings. This is the **detection module** of the Smart Glasses for the Visually Impaired project.

![demo](docs/detection_demo.gif)

## What it does

1. Grabs frames from the camera / video stream
2. Detects people and hazards with **YOLO26n** (Ultralytics)
3. Estimates proximity from bounding-box size and position
4. Speaks short alerts, e.g. *"person, close, on your left"*

Like every module in the project, it follows one convention: **image in, text out.**

```python
from detection import detect

text = detect(frame)   # -> "person, two meters ahead, slightly right"
```

## How alerts work

| Signal | Used for |
|---|---|
| Box area relative to frame | Proximity (far / near / very close) |
| Box center x-position | Direction (left / ahead / right) |
| Class + confidence threshold | What to announce, so low-confidence noise is skipped |
| Cooldown per object | Avoid repeating the same warning every frame |

## Performance

Target: **~30 FPS** for smooth live use. What helps:

- Nano model (`yolo26n`) with a reduced input size (`imgsz=320–416`)
- Run detection on the latest frame only and drop stale frames, so latency doesn't build up
- Run capture, inference, and speech in separate threads, so TTS never blocks the video loop
- Restrict classes to the ones that matter (`classes=[0, ...]`)
- Use half precision on GPU (`half=True`), or export to ONNX / TFLite / NCNN on the Raspberry Pi

| Setup | FPS | Latency |
|---|---|---|
| Colab GPU | – | – |
| Raspberry Pi | – | – |

## Getting Started

```bash
pip install ultralytics opencv-python
```

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.predict(source=0, imgsz=416, conf=0.4, stream=True)
```

Run the full demo notebook: `VisionAid_Final_Presentation.ipynb` (Open in Colab).

## Project Structure

```
detection/
├── detect.py        # detect(frame) -> text
├── proximity.py     # box size/position -> distance + direction
├── speech.py        # non-blocking TTS alerts
├── models/          # weights / exported models
└── notebooks/       # Colab demo
```

## Evaluation

| Metric | Result |
|---|---|
| mAP / detection accuracy | – |
| Distance estimate error | – |
| FPS | – |
| False alert rate | – |

## Limitations

- Proximity from box size is a heuristic and varies with object size and camera angle
- Accuracy drops in low light and with motion blur
- Not a replacement for a cane or guide; use as a supplementary aid

## Future Work

- True depth estimation (monocular depth model or stereo / ToF sensor)
- Tracking to announce only *new* or *approaching* objects
- Quantized models for lower latency on embedded hardware
