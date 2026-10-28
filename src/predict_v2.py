"""Phase 2 prediction: item type -> category, with rejection and B3 caution.

CLI:  python -m src.predict_v2 <image path>
"""
import sys
from dataclasses import dataclass, field

import numpy as np

from src.inference_v2 import predict_arrays
from src.preprocessing import InvalidImageError, load_image_bgr, validate_image
from src.taxonomy import CATEGORIES, CATEGORY_NAMES, ITEM_TO_CATEGORY, item_key
from src.v2_config import B3_CAUTION_THRESHOLD, CATEGORY_REJECT_THRESHOLD


@dataclass
class PredictionV2:
    category: str
    category_confidence: float
    item: str
    item_confidence: float
    category_probs: dict = field(default_factory=dict)
    item_probs: dict = field(default_factory=dict)
    accepted: bool = True
    b3_caution: bool = False
    message: str = ""


def predict_bgr_v2(img_bgr, reject_threshold=None, b3_threshold=None) -> PredictionV2:
    """Classify a BGR image. Raises InvalidImageError for unusable images."""
    reject_threshold = CATEGORY_REJECT_THRESHOLD if reject_threshold is None else reject_threshold
    b3_threshold = B3_CAUTION_THRESHOLD if b3_threshold is None else b3_threshold

    classes, p_item, p_cat = predict_arrays(img_bgr)
    top = int(np.argmax(p_cat))
    cat, cat_conf = CATEGORIES[top], float(p_cat[top])
    # Most likely item type INSIDE the winning category, so item and category agree.
    in_cat = [i for i, c in enumerate(classes) if ITEM_TO_CATEGORY[item_key(c)][0] == cat]
    best = max(in_cat, key=lambda i: p_item[i])
    p_b3 = float(p_cat[CATEGORIES.index("b3")])

    pred = PredictionV2(
        category=cat, category_confidence=cat_conf,
        item=classes[best], item_confidence=float(p_item[best]),
        category_probs={c: float(p_cat[i]) for i, c in enumerate(CATEGORIES)},
        item_probs={c: float(p_item[i]) for i, c in enumerate(classes)},
        accepted=cat_conf >= reject_threshold,
        b3_caution=p_b3 >= b3_threshold,
    )
    if pred.accepted:
        pred.message = f"Prediksi diterima: {CATEGORY_NAMES[cat]}."
    else:
        pred.message = (
            f"Keyakinan model rendah ({cat_conf:.0%} < {reject_threshold:.0%}). Item mungkin "
            "di luar jenis yang dikenali, atau fotonya kurang jelas. Coba foto lain, atau "
            "pilih kategori secara manual."
        )
    if pred.b3_caution and cat != "b3":
        pred.message += (f" PERINGATAN: ada kemungkinan {p_b3:.0%} bahwa ini sampah B3. "
                         "Jika ragu, perlakukan sebagai B3.")
    return pred


def predict_path_v2(path, **kw) -> PredictionV2:
    img = load_image_bgr(path)
    validate_image(img)
    return predict_bgr_v2(img, **kw)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict_v2 <image path>")
        return 2
    try:
        p = predict_path_v2(sys.argv[1])
    except InvalidImageError as e:
        print(f"INVALID IMAGE: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"Category  : {CATEGORY_NAMES[p.category]} (confidence {p.category_confidence:.1%})")
    print(f"Item type : {p.item} ({p.item_confidence:.1%} of the item probability)")
    print("Category probabilities:", {k: round(v, 3) for k, v in p.category_probs.items()})
    print(f"Status    : {'ACCEPTED' if p.accepted else 'REJECTED'} | B3 caution: {p.b3_caution}")
    print(f"Message   : {p.message}")
    print("Confidence is the model's own score, not the probability that it is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())