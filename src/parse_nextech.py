"""Phase 2, Step 1b: derive item class and source-image id from NexTech14 file names.

Needs outputs/v2/nextech_files.csv (from  python -m src.inspect_nextech).
Run:  python -m src.parse_nextech

File name pattern seen in the data:  <item>-<number>[_aug<index>].jpg
  e.g. battery-10_aug1764.jpg  ->  item "battery", source image 10, augmented copy.
The pattern is an assumption: unmatched names are reported, never guessed.
"""
import re
import sys
from pathlib import PurePosixPath

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import SEED
from src.taxonomy import ITEM_TO_CATEGORY, item_key
from src.v2_config import NEXTECH_DIR, OUTPUTS_V2

PATTERN = re.compile(r"^(?P<name>.+?)-(?P<num>\d+)(?:_aug(?P<aug>\d+))?$", re.IGNORECASE)


def main() -> int:
    csv = OUTPUTS_V2 / "nextech_files.csv"
    if not csv.exists():
        print(f"ERROR: {csv} not found. Run  py -m src.inspect_nextech  first.")
        return 1
    df = pd.read_csv(csv)
    df = df[df["readable"]].copy()

    matches = df["path"].map(lambda p: PATTERN.match(PurePosixPath(p).stem))
    df["matched"] = matches.map(lambda m: m is not None)

    print(f"Readable images: {len(df)} | file names matching the pattern: {int(df['matched'].sum())}")
    bad = df[~df["matched"]]
    print(f"Names NOT matching: {len(bad)}")
    for p in bad["path"].head(10):
        print("  -", p)

    ok = df[df["matched"]].copy()
    mm = matches[ok.index]
    ok["fine_class"] = mm.map(lambda m: m.group("name").lower())
    ok["orig_num"] = mm.map(lambda m: int(m.group("num")))
    ok["is_aug"] = mm.map(lambda m: m.group("aug") is not None)
    ok["item_key"] = ok["fine_class"].map(item_key)
    ok["category"] = ok["item_key"].map(
        lambda k: ITEM_TO_CATEGORY[k][0] if k in ITEM_TO_CATEGORY else "(UNMAPPED)")
    ok["group_key"] = ok["item_key"] + "-" + ok["orig_num"].astype(str)

    print("\n=== A. Item class (from file name) x parent folder ===")
    print(pd.crosstab(ok["fine_class"], ok["cls"], margins=True).to_string())

    multi = ok.groupby("fine_class")["cls"].nunique()
    multi = multi[multi > 1]
    print(f"\nItem classes found in more than one parent folder: "
          f"{list(multi.index) if len(multi) else 'none'}")

    unmapped = sorted(ok.loc[ok["category"] == "(UNMAPPED)", "fine_class"].unique())
    print(f"Item classes without a category in taxonomy.py: {unmapped if unmapped else 'none'}")

    print("\n=== B. Per item: images, distinct source images, augmented vs not ===")
    per = ok.groupby("fine_class").agg(
        images=("path", "size"),
        source_images=("group_key", "nunique"),
        augmented=("is_aug", "sum"),
    )
    per["not_augmented"] = per["images"] - per["augmented"]
    per["images_per_source"] = (per["images"] / per["source_images"]).round(1)
    per["category"] = [ok.loc[ok["fine_class"] == c, "category"].iloc[0] for c in per.index]
    print(per.to_string())
    print(f"\nTotal distinct source images: {ok['group_key'].nunique()} "
          f"(of {len(ok)} files)")

    print("\n=== C. Per category: images and distinct source images ===")
    cat = ok.groupby("category").agg(images=("path", "size"),
                                     source_images=("group_key", "nunique"))
    print(cat.to_string())

    print("\n=== D. Leakage across the dataset's OWN split folders (by source-image id) ===")
    spans = ok.groupby("group_key")["folder_split"].nunique()
    multi_groups = spans[spans > 1].index
    print(f"  source images appearing in more than one split folder: "
          f"{len(multi_groups)} of {len(spans)}")
    train_keys = set(ok.loc[ok["folder_split"] == "train", "group_key"])
    for s in ["val", "test"]:
        sub = ok[ok["folder_split"] == s]
        leaked = sub["group_key"].isin(train_keys)
        print(f"  {s} images whose source image also has a copy in train: "
              f"{int(leaked.sum())} of {len(sub)} ({leaked.mean():.1%})")

    keep = ["path", "folder_split", "cls", "fine_class", "item_key", "category",
            "group_key", "is_aug", "md5", "dhash"]
    ok[keep].to_csv(OUTPUTS_V2 / "nextech_parsed.csv", index=False)
    per.to_csv(OUTPUTS_V2 / "nextech_per_item.csv")

    # Visual check that images sharing a group_key really come from one source image.
    big = ok.groupby("group_key").size()
    big = big[big >= 6].index.tolist()
    if big:
        rng = np.random.default_rng(SEED)
        picks = rng.choice(big, size=min(4, len(big)), replace=False)
        fig, axes = plt.subplots(len(picks), 6, figsize=(13, 2.3 * len(picks)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for r, g in enumerate(picks):
            for c, (_, row) in enumerate(ok[ok["group_key"] == g].head(6).iterrows()):
                img = cv2.imdecode(np.fromfile(str(NEXTECH_DIR / row["path"]), dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                axes[r, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[r, c].set_title(f"{g} [{row['folder_split']}]", fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUTS_V2 / "group_examples.png", dpi=100)
        plt.close(fig)
        print("\nSaved group_examples.png (open it: do images in a row show the same object?)")

    print(f"\nSaved to {OUTPUTS_V2}: nextech_parsed.csv, nextech_per_item.csv")
    return 0 if len(bad) == 0 and not unmapped else 1


if __name__ == "__main__":
    sys.exit(main())