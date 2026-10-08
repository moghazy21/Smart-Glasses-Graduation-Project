# Smart Glasses for the Visually Impaired

> A low-cost, voice-controlled wearable assistant that describes the world through audio: it detects obstacles, reads text, recognizes faces and banknotes, and summarizes documents.

![demo](docs/demo.gif)

**Graduation Project** · Badr University in Cairo · Supervisor: Dr. ______

## Why

Visually impaired people can't easily read signs or screens, tell who is approaching, handle cash, or judge obstacle distance. Commercial devices are expensive, closed-source, and often need constant internet or a human operator. This project aims for an **affordable, hands-free, mostly offline** alternative.

## Features

| Say | The glasses |
|---|---|
| "What's in front of me?" | Detect objects and estimate distance ("person, two meters ahead") |
| "Read this" | OCR on signs, books, and screens, ignoring small background text |
| "Who is this?" | Recognize registered faces; unknown faces are reported as unrecognized |
| "What money is this?" | Classify Egyptian banknote denominations |
| "Summarize this" | Summarize a document and read the summary aloud |

All output is spoken through a bone-conduction / open-ear speaker so the user can still hear their surroundings.

## Architecture

```
Voice command → Voice Assistant → capture frame → module → text → TTS → audio
```

Four independent modules share one convention: **take image / audio / text, return text.**

| Module | Responsibility |
|---|---|
| `detection/` | Object detection (YOLO nano/small) + depth estimation |
| `ocr_face/` | EasyOCR with area/confidence filtering, face recognition, currency classifier |
| `summarize_tts/` | PDF text extraction (+ OCR for scans), LLM summarization, Piper TTS |
| `assistant/` | Speech-to-text, command routing, interaction flow |

## Tech Stack

- **Language:** Python (prototyping in Google Colab)
- **Vision:** OpenCV, PyTorch / Ultralytics YOLO, EasyOCR, dlib `face_recognition` or InsightFace
- **Currency:** MobileNet transfer learning on a custom banknote dataset
- **Speech/Language:** STT engine, Piper TTS, language model for summaries
- **Hardware:** Raspberry Pi-class board, camera module, microphone, bone-conduction speaker, battery

## Getting Started

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
python main.py
```

Register a face:

```bash
python ocr_face/register_face.py --name "Ahmed" --images data/faces/ahmed/
```

## Project Structure

```
├── assistant/        # voice commands + routing
├── detection/        # object detection + depth
├── ocr_face/         # OCR, faces, currency
├── summarize_tts/    # summarization + speech
├── data/             # sample images, face DB (git-ignored)
├── models/           # weights
├── docs/             # diagrams, proposal, demo
└── main.py
```

## Evaluation

| Metric | Result |
|---|---|
| Detection accuracy / distance error | – |
| OCR accuracy (signs, books, screens) | – |
| Face true/false match rate | – |
| Currency accuracy per denomination | – |
| End-to-end latency (voice → speech) | – |

## Privacy

Face data is stored **locally**, only for contacts the user registers. Nothing is uploaded to the cloud.

## Limitations & Future Work

- Accuracy drops in poor lighting and extreme angles
- Latency on embedded hardware
- Planned: navigation, more languages, more currencies

## License

MIT

## Disclaimer

Research and educational prototype. Not a certified safety or medical device; don't rely on it as a sole aid for navigation.
