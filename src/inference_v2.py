"""Phase 2 inference helpers: item probabilities and category probabilities."""
import numpy as np
import torch

from src.model_v2 import load_model_v2
from src.preprocessing import to_model_input, validate_image
from src.taxonomy import CATEGORIES, ITEM_TO_CATEGORY, item_key
from src.v2_config import MODEL_V2_PATH

_STATE = {}


def _load():
    if not _STATE:
        if not MODEL_V2_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_V2_PATH}. Train it first (python -m src.train_v2).")
        model, ckpt = load_model_v2(MODEL_V2_PATH)
        classes = list(ckpt["classes"])
        m = np.zeros((len(classes), len(CATEGORIES)), dtype=np.float64)
        for i, c in enumerate(classes):
            m[i, CATEGORIES.index(ITEM_TO_CATEGORY[item_key(c)][0])] = 1.0
        _STATE.update(model=model, classes=classes, cat_matrix=m)
    return _STATE


def predict_arrays(img_bgr):
    """Returns (item_classes, item_probs, category_probs) for one BGR image."""
    validate_image(img_bgr)
    s = _load()
    x = torch.from_numpy(to_model_input(img_bgr)).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(s["model"](x), dim=1)[0].numpy().astype(np.float64)
    return s["classes"], p, p @ s["cat_matrix"]