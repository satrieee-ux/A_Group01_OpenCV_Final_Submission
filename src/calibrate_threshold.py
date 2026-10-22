"""Stage 5: how does the confidence threshold trade coverage against accuracy?

Uses ONLY the validation split. The test split is never loaded.
Run:  python -m src.calibrate_threshold
"""
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROJECT_ROOT, SPLIT_CSV, TARGET_CLASSES
from src.predict import predict_probs
from src.preprocessing import load_image_bgr


def main():
    df = pd.read_csv(SPLIT_CSV)
    val = df[df["split"] == "val"]
    confs, correct = [], []
    for _, r in val.iterrows():
        probs = predict_probs(load_image_bgr(PROJECT_ROOT / r["path"]))
        top = int(np.argmax(probs))
        confs.append(float(probs[top]))
        correct.append(TARGET_CLASSES[top] == r["label"])
    confs, correct = np.array(confs), np.array(correct)

    print(f"Validation images: {len(val)} | overall accuracy: {correct.mean():.4f}")
    print(f"Mean confidence - correct: {confs[correct].mean():.3f} | "
          f"wrong: {confs[~correct].mean():.3f} (n wrong = {(~correct).sum()})\n")
    print(f"{'threshold':>9} {'accepted':>9} {'rejected':>9} {'coverage':>9} {'acc(accepted)':>14}")
    rows = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m = confs >= t
        acc = correct[m].mean() if m.any() else float("nan")
        rows.append({"threshold": t, "accepted": int(m.sum()), "rejected": int((~m).sum()),
                     "coverage": m.mean(), "accuracy_on_accepted": acc})
        print(f"{t:>9.2f} {m.sum():>9d} {(~m).sum():>9d} {m.mean():>9.1%} {acc:>14.4f}")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "threshold_calibration.csv", index=False)
    print(f"\nSaved: {OUTPUTS_DIR / 'threshold_calibration.csv'}")


if __name__ == "__main__":
    main()