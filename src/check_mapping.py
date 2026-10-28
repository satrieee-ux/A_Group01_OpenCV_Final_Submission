"""Phase 2, Step 2: check that every NexTech14 class folder has a category.

Needs outputs/v2/nextech_files.csv (made by:  python -m src.inspect_nextech).
Run:  python -m src.check_mapping
"""
import sys

import pandas as pd

from src.taxonomy import CATEGORIES, ITEM_TO_CATEGORY, item_key
from src.v2_config import OUTPUTS_V2


def main() -> int:
    csv = OUTPUTS_V2 / "nextech_files.csv"
    if not csv.exists():
        print(f"ERROR: {csv} not found. Run  py -m src.inspect_nextech  first.")
        return 1
    df = pd.read_csv(csv)
    df = df[df["readable"]]
    counts = df.groupby("cls").size().sort_index()

    rows, unmapped = [], []
    for cls, n in counts.items():
        key = item_key(cls)
        if key in ITEM_TO_CATEGORY:
            rows.append({"folder": cls, "item_key": key,
                         "category": ITEM_TO_CATEGORY[key][0], "images": int(n)})
        else:
            unmapped.append((cls, int(n)))
            rows.append({"folder": cls, "item_key": key, "category": "(UNMAPPED)",
                         "images": int(n)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\nImages per category:")
    per_cat = out.groupby("category")["images"].sum()
    for c in CATEGORIES + ["(UNMAPPED)"]:
        if c in per_cat.index:
            print(f"  {c:<12}{int(per_cat[c]):>7}")
    print(f"  {'TOTAL':<12}{int(out['images'].sum()):>7}")

    unused = [k for k in ITEM_TO_CATEGORY if k not in set(out["item_key"])]
    if unused:
        print("\nDefined in taxonomy but NOT found in the dataset:", unused)
    out.to_csv(OUTPUTS_V2 / "mapping_check.csv", index=False)
    print(f"\nSaved: {OUTPUTS_V2 / 'mapping_check.csv'}")

    if unmapped:
        print("\nERROR: unmapped folders:", unmapped)
        print("Fix src/taxonomy.py (ALIASES or ITEM_TO_CATEGORY), then run again.")
        return 1
    print("\nOK: every folder has exactly one category.")
    return 0


if __name__ == "__main__":
    sys.exit(main())