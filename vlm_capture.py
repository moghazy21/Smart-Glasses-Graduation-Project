"""
Minimal version: capture a webcam image and describe it with SmolVLM.

ENTER      -> capture + describe
q + ENTER  -> quit
"""

import time

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- config ----------------
CAMERA_INDEX = 0
VLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_SIZE = 512
MAX_NEW_TOKENS = 100
PROMPT = "What do you see in this image? List the main objects and say whether they are on the left, right, or in the center."
SAVE_CAPTURES = True          # saves each capture as capture_N.jpg
# ----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading {VLM_ID} on {device} ...")
processor = AutoProcessor.from_pretrained(VLM_ID)
model = AutoModelForImageTextToText.from_pretrained(VLM_ID, torch_dtype=dtype).to(device)
model.eval()


def capture_image(cap):
    frame = None
    for _ in range(5):                    # skip stale buffered frames
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.03)
    if frame is None:
        raise RuntimeError("Could not read from the camera. Try another CAMERA_INDEX.")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
    return image


def describe(image, prompt=PROMPT):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: (v.to(dtype) if v.dtype.is_floating_point else v) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")

    print("\nReady. ENTER = capture, q + ENTER = quit.\n")
    n = 0
    try:
        while True:
            if input().strip().lower() == "q":
                break
            t0 = time.time()
            image = capture_image(cap)
            if SAVE_CAPTURES:
                image.save(f"capture_{n}.jpg")
                n += 1
            print(f"[{time.time() - t0:.1f}s] {describe(image)}\n")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
