"""Phase 2, Step 3: split NexTech14 by SOURCE IMAGE (not by file).

Run:  python -m src.prepare_nextech        (option: --threshold 4)
Train = all files (originals + augmented copies) of the train sources.
Val/test = ORIGINAL (non-augmented) files only of the val/test sources.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED
from src.prepare_data import group_near_duplicates, hamming_matrix
from src.v2_config import OUTPUTS_V2, SPLIT_V2_CSV

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def to_u64(series):
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


def min_dist_to(train_hashes, h):
    x = np.bitwise_xor(train_hashes, h).view(np.uint8).reshape(-1, 8)
    return int(POP[x].sum(axis=1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=4,
                    help="max hash distance (of 64 bits) to merge two source images")
    ap.add_argument("--keep-aug-in-eval", action="store_true",
                    help="keep augmented copies in val/test (not recommended)")
    args = ap.parse_args()

    src = OUTPUTS_V2 / "nextech_parsed.csv"
    if not src.exists():
        print(f"ERROR: {src} not found. Run  py -m src.parse_nextech  first.")
        return 1
    df = pd.read_csv(src, dtype={"dhash": str, "md5": str})
    df = df.sort_values("path").reset_index(drop=True)

    orig = df[~df["is_aug"]].copy().reset_index(drop=True)
    per_src = orig.groupby("group_key").size()
    missing = set(df["group_key"]) - set(orig["group_key"])
    print(f"Source images (originals): {len(orig)} | ids with exactly 1 original: "
          f"{int((per_src == 1).sum())} of {len(per_src)} | ids without an original: {len(missing)}")
    if missing or (per_src != 1).any():
        print("ERROR: source-id assumption violated. Send this output to the mentor.")
        return 1

    hashes = to_u64(orig["dhash"])
    labels = orig["fine_class"].to_numpy()
    dist = hamming_matrix(hashes)
    same = labels[:, None] == labels[None, :]
    roots = group_near_duplicates(dist, same, args.threshold)
    orig["group_id"] = [f"g{int(r):05d}" for r in roots]
    sizes = orig.groupby("group_id").size()
    print(f"Near-duplicate merge (distance <= {args.threshold}): {len(orig)} source images -> "
          f"{sizes.size} groups | groups with >1 source: {int((sizes > 1).sum())} | "
          f"largest: {int(sizes.max())}")
    df["group_id"] = df["group_key"].map(dict(zip(orig["group_key"], orig["group_id"])))

    gtab = orig.groupby("group_id").agg(
        fine_class=("fine_class", "first"), category=("category", "first")).reset_index()
    trainval, test = train_test_split(
        gtab, test_size=0.15, stratify=gtab["fine_class"], random_state=SEED)
    train, val = train_test_split(
        trainval, test_size=0.15 / 0.85, stratify=trainval["fine_class"], random_state=SEED)
    split_of = {**{g: "train" for g in train["group_id"]},
                **{g: "val" for g in val["group_id"]},
                **{g: "test" for g in test["group_id"]}}
    df["split"] = df["group_id"].map(split_of)

    if not args.keep_aug_in_eval:
        drop = (df["split"] != "train") & df["is_aug"]
        print(f"Augmented copies removed from val/test: {int(drop.sum())}")
        df = df[~drop]
    train_md5 = set(df.loc[df["split"] == "train", "md5"])
    dup = (df["split"] != "train") & df["md5"].isin(train_md5)
    print(f"Val/test files identical to a train file (removed): {int(dup.sum())}")
    df = df[~dup]

    originals = df[~df["is_aug"]]
    print("\nSource images (originals) per item class and split:")
    t1 = pd.crosstab(originals["fine_class"], originals["split"], margins=True)
    print(t1[["train", "val", "test", "All"]].to_string())
    print("\nSource images per category and split:")
    t2 = pd.crosstab(originals["category"], originals["split"], margins=True)
    print(t2[["train", "val", "test", "All"]].to_string())
    print("\nFiles used per category and split (train includes augmented copies):")
    t3 = pd.crosstab(df["category"], df["split"], margins=True)
    print(t3[["train", "val", "test", "All"]].to_string())

    print("\nLeakage checks:")
    print(f"  groups spanning more than one split: "
          f"{int((df.groupby('group_id')['split'].nunique() > 1).sum())} (must be 0)")
    print(f"  source ids spanning more than one split: "
          f"{int((df.groupby('group_key')['split'].nunique() > 1).sum())} (must be 0)")
    print(f"  identical files in more than one split: "
          f"{int((df.groupby('md5')['split'].nunique() > 1).sum())} (must be 0)")
    tr_h = to_u64(df.loc[df["split"] == "train", "dhash"])
    for s in ["val", "test"]:
        eh = to_u64(df.loc[df["split"] == s, "dhash"])
        d = np.array([min_dist_to(tr_h, h) for h in eh])
        print(f"  {s}: images with a train image at hash distance 0: {int((d == 0).sum())}; "
              f"<= {args.threshold}: {int((d <= args.threshold).sum())} of {len(eh)}")
    print("  (the hash misses flipped/rotated copies, so this audit is a lower bound)")

    small = [c for c in t1.index if c != "All" and
             (t1.loc[c, "val"] < 10 or t1.loc[c, "test"] < 10)]
    print(f"\nItem classes with fewer than 10 val or test images: {small if small else 'none'}")

    SPLIT_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["path", "split", "fine_class", "category", "group_id", "group_key",
        "is_aug", "md5"]].to_csv(SPLIT_V2_CSV, index=False)
    t1.to_csv(OUTPUTS_V2 / "split_v2_sources_per_item.csv")
    t3.to_csv(OUTPUTS_V2 / "split_v2_files_per_category.csv")
    print(f"\nSaved: {SPLIT_V2_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())