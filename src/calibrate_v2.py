"""Phase 2, Step 5: confidence calibration on the VALIDATION split only.

Run:  python -m src.calibrate_v2
The test split is never loaded.
"""
import math

import numpy as np
import pandas as pd

from src.inference_v2 import predict_arrays
from src.preprocessing import load_image_bgr
from src.taxonomy import CATEGORIES
from src.v2_config import NEXTECH_DIR, OUTPUTS_V2, SPLIT_V2_CSV


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    df = pd.read_csv(SPLIT_V2_CSV)
    val = df[df["split"] == "val"].reset_index(drop=True)
    rows = []
    for _, r in val.iterrows():
        img = load_image_bgr(NEXTECH_DIR / r["path"])
        classes, p_item, p_cat = predict_arrays(img)
        top_cat = int(np.argmax(p_cat))
        rows.append({
            "path": r["path"], "fine_class": r["fine_class"], "category": r["category"],
            "pred_item": classes[int(np.argmax(p_item))],
            "pred_category": CATEGORIES[top_cat], "cat_conf": float(p_cat[top_cat]),
            **{f"p_{c}": float(p_cat[i]) for i, c in enumerate(CATEGORIES)},
        })
    res = pd.DataFrame(rows)
    res["cat_correct"] = res["category"] == res["pred_category"]
    res["item_correct"] = res["fine_class"] == res["pred_item"]
    conf = res["cat_conf"].to_numpy()
    ok = res["cat_correct"].to_numpy()
    n = len(res)

    print(f"Validation images: {n} | item accuracy: {res['item_correct'].mean():.4f} | "
          f"category accuracy: {ok.mean():.4f} ({int(ok.sum())}/{n})")
    print("Validation images per category:", res["category"].value_counts().to_dict())
    if (~ok).any():
        print(f"Mean category confidence - correct: {conf[ok].mean():.3f} | "
              f"wrong: {conf[~ok].mean():.3f} (n wrong = {int((~ok).sum())})")
    print("\nWrong category predictions (val):")
    for _, r in res[~res["cat_correct"]].sort_values("cat_conf", ascending=False).iterrows():
        print(f"  {r['path']} | true={r['category']} ({r['fine_class']}) | "
              f"pred={r['pred_category']} ({r['pred_item']}) | conf={r['cat_conf']:.3f}")

    print("\n=== A. Category rejection threshold (confidence of top category) ===")
    print(f"{'threshold':>9} {'accepted':>9} {'rejected':>9} {'coverage':>9} "
          f"{'acc(accepted)':>14} {'wrong kept':>11} {'correct rejected':>17}")
    rows_a = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        m = conf >= t
        acc = ok[m].mean() if m.any() else float("nan")
        rows_a.append({"threshold": t, "accepted": int(m.sum()), "rejected": int((~m).sum()),
                       "coverage": m.mean(), "acc_on_accepted": acc,
                       "wrong_kept": int((m & ~ok).sum()), "correct_rejected": int((~m & ok).sum())})
        print(f"{t:>9.2f} {m.sum():>9d} {(~m).sum():>9d} {m.mean():>9.1%} {acc:>14.4f} "
              f"{int((m & ~ok).sum()):>11d} {int((~m & ok).sum()):>17d}")

    print("\n=== B. Category recall at the top-1 decision (no rejection) ===")
    for c in CATEGORIES:
        sub = res[res["category"] == c]
        k = int(sub["cat_correct"].sum())
        lo, hi = wilson(k, len(sub))
        print(f"  {c:<10} {k}/{len(sub)}  recall {k / max(len(sub), 1):.3f}  "
              f"(95% Wilson CI {lo:.2f}-{hi:.2f})")

    print("\n=== C. B3 caution threshold (flag B3 when P(B3) >= t, even if B3 is not the top category) ===")
    b3 = (res["category"] == "b3").to_numpy()
    pb3 = res["p_b3"].to_numpy()
    n_b3, n_other = int(b3.sum()), int((~b3).sum())
    print(f"True B3 images: {n_b3} | non-B3 images: {n_other}")
    print(f"{'t_b3':>6} {'B3 flagged':>11} {'sensitivity':>12} {'95% CI':>13} "
          f"{'false alarms':>13} {'FA rate':>8}")
    rows_c = []
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        trig = pb3 >= t
        k, fa = int(trig[b3].sum()), int(trig[~b3].sum())
        lo, hi = wilson(k, n_b3)
        rows_c.append({"t_b3": t, "flagged": k, "n_b3": n_b3, "sensitivity": k / max(n_b3, 1),
                       "ci_low": lo, "ci_high": hi, "false_alarms": fa,
                       "false_alarm_rate": fa / max(n_other, 1)})
        print(f"{t:>6.2f} {k:>5d}/{n_b3:<5d} {k / max(n_b3, 1):>12.3f} {lo:>6.2f}-{hi:<6.2f} "
              f"{fa:>7d}/{n_other:<5d} {fa / max(n_other, 1):>8.1%}")
    print("\nLowest P(B3) among true B3 images:")
    for _, r in res[res["category"] == "b3"].nsmallest(3, "p_b3").iterrows():
        print(f"  {r['path']} | P(B3)={r['p_b3']:.3f} | pred={r['pred_category']}")
    print("Highest P(B3) among NON-B3 images:")
    for _, r in res[res["category"] != "b3"].nlargest(5, "p_b3").iterrows():
        print(f"  {r['path']} | true={r['category']} ({r['fine_class']}) | P(B3)={r['p_b3']:.3f}")

    print("\nNote: val is tiny (B3 n=%d). These tables suggest ranges, not precise optima." % n_b3)
    OUTPUTS_V2.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUTS_V2 / "calibration_val_v2.csv", index=False)
    pd.DataFrame(rows_a).to_csv(OUTPUTS_V2 / "calibration_table_category.csv", index=False)
    pd.DataFrame(rows_c).to_csv(OUTPUTS_V2 / "calibration_table_b3.csv", index=False)
    print(f"Saved to {OUTPUTS_V2}")


if __name__ == "__main__":
    main()