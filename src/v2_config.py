"""Paths and locked thresholds for Phase 2 (v2)."""
from src.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR

NEXTECH_DIR = DATA_DIR / "raw" / "nextech14"
OUTPUTS_V2 = OUTPUTS_DIR / "v2"
SPLIT_V2_CSV = SPLITS_DIR / "split_v2.csv"
MODEL_V2_PATH = MODELS_DIR / "waste_classifier_v2.pt"

# Thresholds chosen from VALIDATION calibration (src/calibrate_v2.py) and
# locked BEFORE the test split is evaluated. Locked on: 28/10/2026.
CATEGORY_REJECT_THRESHOLD = 0.80
B3_CAUTION_THRESHOLD = 0.30