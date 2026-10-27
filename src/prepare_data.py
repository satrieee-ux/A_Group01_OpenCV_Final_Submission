"""Stage 3: validate images, group near-duplicates, create a reproducible split.

Run from the project root:  python -m src.prepare_data
Optional: python -m src.prepare_data --threshold 10
"""
import argparse
import hashlib

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    IMAGE_EXTENSIONS, OUTPUTS_DIR, PROJECT_ROOT, SEED, SOURCE_TO_TARGET,
    SPLIT_CSV, SPLITS_DIR,
)
from src.inspect_dataset import find_dataset_root
from src.preprocessing import load_image_bgr

AUDIT_THRESHOLD = 16  # looser threshold used only to audit remaining similarity
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash64(img_bgr) -> int:
    """64-bit difference hash: similar-looking images give similar hashes."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def collect_records(root):
    records, unreadable = [], []
    for source_name, target in SOURCE_TO_TARGET.items():
        class_dir = next(
            (p for p in root.iterdir() if p.is_dir() and p.name.lower() == source_name), None
        )
        if class_dir is None:
            raise SystemExit(f"ERROR: class folder '{source_name}' not found in {root}")
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = load_image_bgr(f)
            if img is None:
                unreadable.append(str(f))
                continue
            records.append({
                "path": f.relative_to(PROJECT_ROOT).as_posix(),
                "label": target,
                "md5": hashlib.md5(f.read_bytes()).hexdigest(),
                "dhash": dhash64(img),
            })
    return records, unreadable


def hamming_matrix(hashes):
    n = len(hashes)
    dist = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        x = np.bitwise_xor(hashes, hashes[i]).view(np.uint8).reshape(n, 8)
        dist[i] = _POPCOUNT[x].sum(axis=1)
    return dist


def group_near_duplicates(dist, same_label, threshold):
    """Union-find: images with hash distance <= threshold (same label) share a group."""
    n = dist.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.where(np.triu((dist <= threshold) & same_label, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def make_split(labels, groups, seed):
    """Approx. 72/14/14 train/val/test, stratified by class, grouped by near-duplicate group."""
    n = len(labels)
    idx = np.arange(n)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(idx, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(
        inner.split(trainval_idx, labels[trainval_idx], groups[trainval_idx])
    )
    split = np.empty(n, dtype=object)
    split[test_idx] = "test"
    split[trainval_idx[val_rel]] = "val"
    split[trainval_idx[tr_rel]] = "train"
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=4,
                        help="max Hamming distance (of 64 bits) to treat images as near-duplicates")
    args = parser.parse_args()

    root = find_dataset_root()
    if root is None:
        raise SystemExit("ERROR: dataset not found. Run Stage 2 first.")

    print("Reading images and computing hashes ...")
    records, unreadable = collect_records(root)
    print(f"Readable images: {len(records)} | unreadable (skipped): {len(unreadable)}")

    df = pd.DataFrame(records)
    hashes = np.array(df["dhash"].tolist(), dtype=np.uint64)
    labels = df["label"].to_numpy()
    same_label = labels[:, None] == labels[None, :]

    dist = hamming_matrix(hashes)
    cross_label_pairs = int(np.triu((dist <= args.threshold) & ~same_label, k=1).sum())

    groups = group_near_duplicates(dist, same_label, args.threshold)
    df["group_id"] = groups
    sizes = pd.Series(groups).value_counts()
    print(f"\nNear-duplicate grouping (threshold <= {args.threshold} of 64 bits):")
    print(f"  groups: {sizes.size} | groups with >1 image: {(sizes > 1).sum()} "
          f"| largest group: {sizes.iloc[0]} images")
    print(f"  exact duplicate files (same MD5): {int(df['md5'].duplicated().sum())}")
    print(f"  near-duplicate pairs with DIFFERENT labels: {cross_label_pairs}")

    df["split"] = make_split(labels, groups, SEED)

    # Hard check: no group may appear in more than one split.
    leaked = df.groupby("group_id")["split"].nunique().gt(1).sum()
    assert leaked == 0, f"{leaked} groups appear in more than one split!"

    # Audit with a looser threshold: similar pairs that still cross splits.
    splits = df["split"].to_numpy()
    cross = (dist <= AUDIT_THRESHOLD) & same_label & (splits[:, None] != splits[None, :])
    audit_pairs = int(np.triu(cross, k=1).sum())
    md5_cross = int(df.groupby("md5")["split"].nunique().gt(1).sum())

    summary = pd.crosstab(df["label"], df["split"], margins=True)[["train", "val", "test", "All"]]
    print("\nImages per class and split:")
    print(summary.to_string())
    print("\nShare of each split (%):")
    print((summary.loc["All", ["train", "val", "test"]] / len(df) * 100).round(1).to_string())

    print("\nLeakage checks:")
    print(f"  groups spanning multiple splits: {leaked} (must be 0)")
    print(f"  identical files in multiple splits: {md5_cross} (must be 0)")
    print(f"  similar pairs (distance <= {AUDIT_THRESHOLD}) still crossing splits: {audit_pairs}")
    print("  Note: perceptual hashes catch near-identical shots, not the same object")
    print("  photographed from a very different angle. Some leakage may remain.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = df.copy()
    out["dhash"] = out["dhash"].map(lambda v: f"{v:016x}")
    out[["path", "label", "split", "group_id", "md5", "dhash"]].to_csv(SPLIT_CSV, index=False)
    summary.to_csv(OUTPUTS_DIR / "split_summary.csv")
    print(f"\nSaved: {SPLIT_CSV}")
    print(f"Saved: {OUTPUTS_DIR / 'split_summary.csv'}")


if __name__ == "__main__":
    main()