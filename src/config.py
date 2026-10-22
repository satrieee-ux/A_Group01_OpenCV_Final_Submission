"""Central configuration. All paths are relative to the project root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "trashnet"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# TrashNet label -> our target label.
# Note: TrashNet's "plastic" and "metal" are broader than bottles/cans.
# We treat them as plastic_bottle / metal_can as a documented simplification.
SOURCE_TO_TARGET = {
    "plastic": "plastic_bottle",
    "metal": "metal_can",
    "paper": "paper",
}

# Alphabetical order; the model's output index follows this order.
TARGET_CLASSES = ["metal_can", "paper", "plastic_bottle"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---- Stage 3 additions ----
SEED = 42
SPLIT_CSV = SPLITS_DIR / "split.csv"

IMAGE_SIZE = 224  # model input is IMAGE_SIZE x IMAGE_SIZE
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # required by the pretrained model
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---- Stage 4 additions ----
MODEL_PATH = MODELS_DIR / "waste_classifier.pt"

# ---- Stage 5 additions ----
# Provisional value. It is re-selected from VALIDATION data only
# (see src/calibrate_threshold.py) and then fixed before test evaluation.
CONFIDENCE_THRESHOLD = 0.80

SORTING_GUIDANCE = {
    "plastic_bottle": (
        "Plastic bottle. Empty it, rinse it quickly, and place it in the "
        "plastic / recyclable bin."
    ),
    "metal_can": (
        "Metal can. Empty it and place it in the metal / recyclable bin."
    ),
    "paper": (
        "Paper. Keep it clean and dry and place it in the paper / recyclable bin. "
        "Greasy or heavily soiled paper usually cannot be recycled."
    ),
}
GUIDANCE_DISCLAIMER = (
    "General guidance only. Local recycling rules may differ."
)