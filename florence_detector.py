from typing import List, Dict
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "microsoft/Florence-2-base"


def load_florence(device: str):
    """Load Florence-2 model and processor."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True
    ).to(device).eval()
    # Some transformer versions expect this flag; default to False if missing
    if not hasattr(model, "_supports_sdpa"):
        setattr(model, "_supports_sdpa", False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def detect_objects(image: Image.Image, model, processor, device: str, max_new_tokens: int = 256) -> List[Dict]:
    """
    Run zero-shot detection on an image; returns list of dicts with box, label, score.
    """
    prompt = "<OD>detect obstacles, vehicles, people, debris on or near railway tracks"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    parsed = processor.post_process_generation(
        outputs,
        target_sizes=[image.size[::-1]],  # (h, w)
        task="<OD>"
    )[0]

    boxes = parsed.get("boxes", [])
    labels = parsed.get("labels", [])
    scores = parsed.get("scores", [None] * len(boxes))

    results = []
    for box, label, score in zip(boxes, labels, scores):
        # Florence boxes are XYXY
        x_min, y_min, x_max, y_max = [float(v) for v in box]
        results.append({
            "label": label,
            "box": (int(x_min), int(y_min), int(x_max), int(y_max)),
            "score": float(score) if score is not None else None
        })
    return results
