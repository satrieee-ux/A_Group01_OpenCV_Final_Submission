"""Stage 2: inspect the raw dataset and verify every image can be decoded.

Run from the project root:  python -m src.inspect_dataset
"""
import csv
import sys
from collections import Counter

import cv2
import numpy as np

from src.config import IMAGE_EXTENSIONS, OUTPUTS_DIR, RAW_DIR, SOURCE_TO_TARGET


def find_dataset_root():
    """Return the folder that directly contains the class folders."""
    candidates = [RAW_DIR] + [p for p in RAW_DIR.rglob("*") if p.is_dir()]
    for folder in candidates:
        names = {p.name.lower() for p in folder.iterdir() if p.is_dir()}
        if set(SOURCE_TO_TARGET) <= names:
            return folder
    return None


def read_image(path):
    """Decode an image with OpenCV. np.fromfile is safe for Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> int:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist. Extract the dataset there.")
        return 1

    root = find_dataset_root()
    if root is None:
        print(f"ERROR: could not find folders {sorted(SOURCE_TO_TARGET)} under {RAW_DIR}")
        return 1
    print(f"Dataset root: {root}\n")

    rows = []
    unreadable = []
    sizes = Counter()

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        ok_count = 0
        for f in files:
            img = read_image(f)
            if img is None:
                unreadable.append(str(f))
                continue
            ok_count += 1
            sizes[(img.shape[1], img.shape[0])] += 1  # (width, height)
        used = class_dir.name.lower() in SOURCE_TO_TARGET
        target = SOURCE_TO_TARGET.get(class_dir.name.lower(), "(excluded)")
        rows.append([class_dir.name, target, len(files), ok_count, "yes" if used else "no"])

    print(f"{'folder':<12}{'target label':<18}{'files':>7}{'readable':>10}{'used':>6}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<18}{r[2]:>7}{r[3]:>10}{r[4]:>6}")

    used_total = sum(r[3] for r in rows if r[4] == "yes")
    print(f"\nReadable images in the 3 target classes: {used_total}")
    print(f"Unreadable files: {len(unreadable)}")
    for u in unreadable[:10]:
        print("  -", u)
    print("\nMost common image sizes (width x height):")
    for (w, h), n in sizes.most_common(3):
        print(f"  {w} x {h}: {n} images")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / "dataset_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["folder", "target_label", "files", "readable", "used"])
        writer.writerows(rows)
    print(f"\nSaved: {out}")

    missing = [c for c in SOURCE_TO_TARGET if c not in {r[0].lower() for r in rows}]
    if missing:
        print(f"ERROR: missing classes: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())