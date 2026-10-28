"""Phase 2, Step 6 check: run predict_v2 on a few VALIDATION images and on a few
out-of-scope images. Demonstration only, NOT an evaluation.

Run:  python -m src.check_predictions_v2
"""
import pandas as pd

from src.inspect_dataset import find_dataset_root
from src.predict_v2 import predict_path_v2
from src.v2_config import NEXTECH_DIR, SPLIT_V2_CSV


def show(name, path, true_label):
    try:
        p = predict_path_v2(path)
    except Exception as e:  # printed, never hidden
        print(f"{name:<36} ERROR: {type(e).__name__}: {e}")
        return
    status = "ACCEPTED" if p.accepted else "REJECTED"
    print(f"{name:<36} true={true_label:<11} pred={p.category:<10} "
          f"conf={p.category_confidence:.2f} {status:<8} "
          f"B3-caution={str(p.b3_caution):<5} item={p.item}")


def main():
    df = pd.read_csv(SPLIT_V2_CSV)
    val = df[df["split"] == "val"].sort_values("path")
    print("=== A. VALIDATION images (2 per category; demonstration, not evaluation) ===")
    for cat, g in val.groupby("category"):
        for _, r in g.head(2).iterrows():
            show(r["path"].split("/")[-1], NEXTECH_DIR / r["path"], cat)

    print("\n=== B. Out-of-scope images: TrashNet cardboard (informal check) ===")
    print("Cardboard is not one of the 11 item types; Lampiran A lists it as anorganik.")
    root = find_dataset_root()
    if root is None:
        print("TrashNet folder not found; skipped.")
        return
    folder = next((p for p in root.iterdir() if p.is_dir() and p.name.lower() == "cardboard"), None)
    if folder is None:
        print("cardboard folder not found; skipped.")
        return
    for f in sorted(folder.glob("*.jpg"))[:3]:
        show(f.name, f, "(anorganik)")


if __name__ == "__main__":
    main()