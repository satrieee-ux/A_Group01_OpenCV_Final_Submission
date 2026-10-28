"""Phase 2, Step 1: inspect the NexTech14 dataset (read-only).

Run from the project root:  python -m src.inspect_nextech
It does not assume a folder layout: class = name of the image's parent folder
(or grandparent if the parent is a train/val/test folder).
"""
import argparse
import hashlib
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import IMAGE_EXTENSIONS, SEED
from src.prepare_data import dhash64, group_near_duplicates, hamming_matrix
from src.v2_config import NEXTECH_DIR, OUTPUTS_V2

SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def scan_and_hash():
    rows, other_ext = [], Counter()
    files = sorted(p for p in NEXTECH_DIR.rglob("*") if p.is_file())
    print(f"Files found: {len(files)}")
    for k, p in enumerate(files, 1):
        if k % 1000 == 0:
            print(f"  processed {k}/{len(files)} ...")
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            other_ext[p.suffix.lower() or "(no extension)"] += 1
            continue
        rel = p.relative_to(NEXTECH_DIR)
        parts = [x.lower() for x in rel.parts[:-1]]
        split = next((x for x in parts if x in SPLIT_NAMES), "none")
        parent = parts[-1] if parts else "(root)"
        cls = parent
        if parent in SPLIT_NAMES:
            cls = parts[-2] if len(parts) >= 2 else "(unknown)"
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        row = {"path": rel.as_posix(), "cls": cls, "folder_split": split,
               "md5": hashlib.md5(data.tobytes()).hexdigest(), "readable": img is not None,
               "width": None, "height": None, "dhash": None}
        if img is not None:
            row["height"], row["width"] = img.shape[:2]
            row["dhash"] = dhash64(img)
        rows.append(row)
    return pd.DataFrame(rows), other_ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=4,
                    help="max Hamming distance (of 64 bits) for 'near-duplicate'")
    ap.add_argument("--max-hash-images", type=int, default=20000)
    args = ap.parse_args()

    if not NEXTECH_DIR.exists():
        raise SystemExit(f"ERROR: {NEXTECH_DIR} does not exist. Extract the dataset there.")
    OUTPUTS_V2.mkdir(parents=True, exist_ok=True)

    df, other_ext = scan_and_hash()
    if df.empty:
        raise SystemExit("ERROR: no .jpg/.jpeg/.png images found.")

    print("\n=== 1. Folder structure (top levels, image counts) ===")
    levels = Counter("/".join(Path_parts[:2]) for Path_parts in
                     (p.split("/")[:-1] for p in df["path"]))
    for name, n in sorted(levels.items()):
        print(f"  {name or '(root)'}: {n} images")
    print(f"Non-image files ignored: {dict(other_ext) if other_ext else 'none'}")

    print("\n=== 2. Images per class and folder-split ===")
    table = pd.crosstab(df["cls"], df["folder_split"], margins=True)
    print(table.to_string())
    print(f"\nClasses found: {df['cls'].nunique()} -> {sorted(df['cls'].unique())}")

    bad = df[~df["readable"]]
    print(f"\n=== 3. Readability === unreadable files: {len(bad)}")
    for p in bad["path"].head(10):
        print("  -", p)
    ok = df[df["readable"]].copy()

    print("\n=== 4. Image sizes (width x height), most common ===")
    for (w, h), n in Counter(zip(ok["width"], ok["height"])).most_common(5):
        print(f"  {w} x {h}: {n} images")

    print("\n=== 5. Exact duplicates (same file content) ===")
    extra = int(ok["md5"].duplicated().sum())
    span = int((ok.groupby("md5")["folder_split"].nunique() > 1).sum())
    print(f"  extra copies: {extra} | identical files present in more than one split folder: {span}")

    aug = ok["path"].str.contains("aug", case=False).sum()
    print(f"\n=== 6. Filename hint of augmentation === files with 'aug' in the path: {int(aug)}")

    print(f"\n=== 7. Near-duplicates (perceptual hash distance <= {args.threshold}) ===")
    if len(ok) > args.max_hash_images:
        print(f"  skipped: {len(ok)} images exceed --max-hash-images")
    else:
        hashes = np.array(ok["dhash"].tolist(), dtype=np.uint64)
        labels = ok["cls"].to_numpy()
        splits = ok["folder_split"].to_numpy()
        dist = hamming_matrix(hashes)
        same = labels[:, None] == labels[None, :]
        groups = group_near_duplicates(dist, same, args.threshold)
        sizes = pd.Series(groups).value_counts()
        close = np.triu((dist <= args.threshold) & same, k=1)
        print(f"  similar same-class pairs: {int(close.sum())}")
        print(f"  groups: {sizes.size} | groups with >1 image: {int((sizes > 1).sum())} "
              f"| largest group: {int(sizes.iloc[0])}")
        if len(set(splits)) > 1:
            cross = close & (splits[:, None] != splits[None, :])
            print(f"  similar pairs that cross the dataset's OWN split folders: {int(cross.sum())}")
            ii, jj = np.where(cross)
            for i, j in list(zip(ii, jj))[:3]:
                print(f"    e.g. {ok.iloc[i]['path']}  <->  {ok.iloc[j]['path']}  (distance {dist[i, j]})")
        print("  Note: the hash does not catch flipped/rotated/recolored copies, so this is a lower bound.")

    out = df.copy()
    out["dhash"] = out["dhash"].map(lambda v: f"{int(v):016x}" if pd.notna(v) else "")
    out.to_csv(OUTPUTS_V2 / "nextech_files.csv", index=False)
    table.to_csv(OUTPUTS_V2 / "nextech_summary.csv")

    classes = sorted(ok["cls"].unique())
    fig, axes = plt.subplots(len(classes), 5, figsize=(14, 2.3 * len(classes)), squeeze=False)
    for r, c in enumerate(classes):
        picks = ok[ok["cls"] == c].sample(n=min(5, (ok["cls"] == c).sum()), random_state=SEED)
        for k in range(5):
            axes[r, k].axis("off")
        for k, (_, row) in enumerate(picks.iterrows()):
            img = cv2.imdecode(np.fromfile(str(NEXTECH_DIR / row["path"]), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            axes[r, k].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[r, k].set_title(c, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS_V2 / "nextech_sample_grid.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved to {OUTPUTS_V2}: nextech_files.csv, nextech_summary.csv, nextech_sample_grid.png")


if __name__ == "__main__":
    main()