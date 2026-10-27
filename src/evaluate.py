"""Stage 7: evaluate the locked model and threshold on the held-out test split.

Run once from the project root:  python -m src.evaluate
Do NOT change the model, threshold or split based on these results.
"""
import json
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    CONFIDENCE_THRESHOLD, MODEL_PATH, OUTPUTS_DIR, PROJECT_ROOT, SPLIT_CSV,
    TARGET_CLASSES,
)
from src.inspect_dataset import find_dataset_root
from src.model import load_model
from src.predict import predict_probs
from src.preprocessing import load_image_bgr

OUT_OF_SCOPE_FOLDERS = ["glass", "cardboard", "trash"]


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def evaluate_test():
    df = pd.read_csv(SPLIT_CSV)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test images: {len(test)}")
    rows = []
    for _, r in test.iterrows():
        probs = predict_probs(load_image_bgr(PROJECT_ROOT / r["path"]))
        top = int(np.argmax(probs))
        rows.append({
            "path": r["path"], "true": r["label"], "pred": TARGET_CLASSES[top],
            "confidence": float(probs[top]),
            **{f"p_{c}": float(p) for c, p in zip(TARGET_CLASSES, probs)},
        })
    res = pd.DataFrame(rows)
    res["correct"] = res["true"] == res["pred"]
    res["accepted"] = res["confidence"] >= CONFIDENCE_THRESHOLD
    res.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    n, k = len(res), int(res["correct"].sum())
    lo, hi = wilson(k, n)
    print(f"\n=== A. All test images (no rejection) ===")
    print(f"Accuracy: {k}/{n} = {k / n:.4f}  (95% Wilson CI: {lo:.3f} - {hi:.3f})")
    print("\nPer-class metrics:")
    print(classification_report(res["true"], res["pred"], labels=TARGET_CLASSES,
                                target_names=TARGET_CLASSES, digits=3))
    cm = confusion_matrix(res["true"], res["pred"], labels=TARGET_CLASSES)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES).to_string())

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(TARGET_CLASSES, rotation=20)
    ax.set_yticks(range(3)); ax.set_yticklabels(TARGET_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (n={n})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    acc_m = res[res["accepted"]]
    rej = res[~res["accepted"]]
    print(f"\n=== B. With the locked rejection threshold {CONFIDENCE_THRESHOLD:.2f} ===")
    print(f"Accepted: {len(acc_m)} ({len(acc_m) / n:.1%}) | rejected: {len(rej)}")
    if len(acc_m):
        print(f"Accuracy on accepted: {acc_m['correct'].mean():.4f} "
              f"({int(acc_m['correct'].sum())}/{len(acc_m)})")
        print(f"Wrong predictions that were still accepted: {int((~acc_m['correct']).sum())}")
    print(f"Rejected images that were actually correct: {int(rej['correct'].sum())} "
          f"of {len(rej)} rejected")

    wrong = res[~res["correct"]].sort_values("confidence", ascending=False)
    print(f"\n=== C. Misclassified test images: {len(wrong)} ===")
    for _, r in wrong.iterrows():
        print(f"  {r['path']} | true={r['true']} | pred={r['pred']} | "
              f"conf={r['confidence']:.3f} | {'accepted' if r['accepted'] else 'rejected'}")
    if len(wrong):
        show = wrong.head(12)
        cols = 4
        rows_n = math.ceil(len(show) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, 3.6 * rows_n), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
            img = load_image_bgr(PROJECT_ROOT / r["path"])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"true: {r['true']}\npred: {r['pred']} ({r['confidence']:.0%})",
                         fontsize=9)
        fig.suptitle("Misclassified test images (highest confidence first)")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "misclassified_examples.png", dpi=110)
        plt.close(fig)

    per_class = classification_report(res["true"], res["pred"], labels=TARGET_CLASSES,
                                      target_names=TARGET_CLASSES, output_dict=True)
    metrics = {
        "n_test": n, "n_correct": k, "accuracy": k / n,
        "accuracy_ci95": [lo, hi], "threshold": CONFIDENCE_THRESHOLD,
        "n_accepted": int(len(acc_m)),
        "accuracy_on_accepted": float(acc_m["correct"].mean()) if len(acc_m) else None,
        "per_class": {c: per_class[c] for c in TARGET_CLASSES},
        "confusion_matrix": {"labels": TARGET_CLASSES, "matrix": cm.tolist()},
    }
    (OUTPUTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


def evaluate_out_of_scope():
    """Rough check: images of unsupported classes. Not part of the test split."""
    root = find_dataset_root()
    rows = []
    for folder in OUT_OF_SCOPE_FOLDERS:
        d = next((p for p in root.iterdir() if p.is_dir() and p.name.lower() == folder), None)
        if d is None:
            print(f"  (folder '{folder}' not found, skipped)")
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = load_image_bgr(f)
            if img is None:
                continue
            probs = predict_probs(img)
            top = int(np.argmax(probs))
            rows.append({"folder": folder, "path": f.relative_to(PROJECT_ROOT).as_posix(),
                         "pred": TARGET_CLASSES[top], "confidence": float(probs[top]),
                         "rejected": float(probs[top]) < CONFIDENCE_THRESHOLD})
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS_DIR / "out_of_scope_results.csv", index=False)

    print(f"\n=== D. Unsupported item types (threshold {CONFIDENCE_THRESHOLD:.2f}) ===")
    print("Not an accuracy measure: the correct behaviour here is rejection.")
    print(f"{'folder':<11}{'images':>7}{'rejected':>10}{'wrongly accepted':>18}{'rejection rate':>16}")
    for folder, g in res.groupby("folder"):
        rej = int(g["rejected"].sum())
        print(f"{folder:<11}{len(g):>7}{rej:>10}{len(g) - rej:>18}{rej / len(g):>16.1%}")
    tot_rej = int(res["rejected"].sum())
    print(f"{'ALL':<11}{len(res):>7}{tot_rej:>10}{len(res) - tot_rej:>18}{tot_rej / len(res):>16.1%}")
    acc = res[~res["rejected"]]
    if len(acc):
        print("\nWrongly accepted images were assigned to:")
        print(acc["pred"].value_counts().to_string())


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    _, ckpt = load_model(MODEL_PATH)
    print(f"Model: {MODEL_PATH.name} | best epoch {ckpt['best_epoch']} | "
          f"stored val acc {ckpt['val_acc']:.4f} | threshold {CONFIDENCE_THRESHOLD}")
    evaluate_test()
    evaluate_out_of_scope()
    print("\nSaved to outputs\\: test_predictions.csv, test_metrics.json, "
          "confusion_matrix.png, misclassified_examples.png, out_of_scope_results.csv")


if __name__ == "__main__":
    main()