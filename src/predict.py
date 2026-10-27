"""Stage 5: inference pipeline (OpenCV preprocessing + model).

Command line:  python -m src.predict path\\to\\image.jpg
"""
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import (
    CONFIDENCE_THRESHOLD, GUIDANCE_DISCLAIMER, MODEL_PATH, SORTING_GUIDANCE,
    TARGET_CLASSES,
)
from src.model import load_model
from src.preprocessing import (
    InvalidImageError, load_image_bgr, to_model_input, validate_image,
)

_MODEL = None


def get_model():
    """Load the model once and reuse it. Raises FileNotFoundError if it is missing."""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Train it first (python -m src.train)."
            )
        _MODEL, _ = load_model(MODEL_PATH)
    return _MODEL


@dataclass
class Prediction:
    label: str                      # top-1 class (even if rejected)
    confidence: float               # top-1 softmax probability
    probabilities: dict = field(default_factory=dict)
    accepted: bool = True           # False if confidence < threshold
    guidance: str = ""              # empty when the prediction is rejected
    message: str = ""


def predict_probs(img_bgr) -> np.ndarray:
    """Softmax probabilities for one BGR image, in TARGET_CLASSES order."""
    validate_image(img_bgr)
    x = torch.from_numpy(to_model_input(img_bgr)).unsqueeze(0)
    with torch.no_grad():
        logits = get_model()(x)
    return torch.softmax(logits, dim=1)[0].numpy()


def predict_bgr(img_bgr, threshold: float = None) -> Prediction:
    """Classify a BGR image. Raises InvalidImageError for unusable images."""
    threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold
    probs = predict_probs(img_bgr)
    top = int(np.argmax(probs))
    label, conf = TARGET_CLASSES[top], float(probs[top])
    pred = Prediction(
        label=label, confidence=conf,
        probabilities={c: float(p) for c, p in zip(TARGET_CLASSES, probs)},
    )
    if conf >= threshold:
        pred.guidance = f"{SORTING_GUIDANCE[label]} {GUIDANCE_DISCLAIMER}"
        pred.message = "Prediction accepted."
    else:
        pred.accepted = False
        pred.message = (
            f"Low confidence ({conf:.0%} < {threshold:.0%}). The item may be outside the "
            "supported categories (plastic bottle, metal can, paper), or the photo may be "
            "unclear. Please try another image of a single item on a plain background."
        )
    return pred


def predict_path(path, threshold: float = None) -> Prediction:
    img = load_image_bgr(path)
    validate_image(img)
    return predict_bgr(img, threshold)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image path>")
        return 2
    try:
        p = predict_path(sys.argv[1])
    except InvalidImageError as e:
        print(f"INVALID IMAGE: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"Top class : {p.label}")
    print(f"Confidence: {p.confidence:.1%} (model score, not a guarantee of correctness)")
    print("Probabilities:", {k: round(v, 3) for k, v in p.probabilities.items()})
    print(f"Status    : {'ACCEPTED' if p.accepted else 'REJECTED'} - {p.message}")
    if p.guidance:
        print(f"Guidance  : {p.guidance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())