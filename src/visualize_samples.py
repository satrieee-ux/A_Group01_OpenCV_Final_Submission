"""Stage 3: save a sample grid and a preprocessing example for the report.

Run from the project root:  python -m src.visualize_samples
"""
import cv2
import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUTS_DIR, PROJECT_ROOT, SEED, SPLIT_CSV, TARGET_CLASSES
from src.preprocessing import load_image_bgr, resize_rgb, to_model_input


def main():
    df = pd.read_csv(SPLIT_CSV)
    train = df[df["split"] == "train"]
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 1) Sample grid: 5 random training images per class
    fig, axes = plt.subplots(len(TARGET_CLASSES), 5, figsize=(15, 9))
    for r, label in enumerate(TARGET_CLASSES):
        picks = train[train["label"] == label].sample(n=5, random_state=SEED)
        for c, (_, row) in enumerate(picks.iterrows()):
            img = load_image_bgr(PROJECT_ROOT / row["path"])
            axes[r, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[r, c].set_title(label, fontsize=10)
            axes[r, c].axis("off")
    fig.suptitle("Random training samples per class (TrashNet)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "sample_grid.png", dpi=120)
    plt.close(fig)

    # 2) Preprocessing example on one image
    row = train.iloc[0]
    img = load_image_bgr(PROJECT_ROOT / row["path"])
    resized = resize_rgb(img)
    tensor = to_model_input(img)
    display = tensor.transpose(1, 2, 0)
    display = (display - display.min()) / (display.max() - display.min())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original ({img.shape[1]}x{img.shape[0]})")
    axes[1].imshow(resized)
    axes[1].set_title(f"BGR->RGB + resize ({resized.shape[1]}x{resized.shape[0]})")
    axes[2].imshow(display)
    axes[2].set_title("Normalised (rescaled for display)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "preprocessing_example.png", dpi=120)
    plt.close(fig)

    print(f"Example image: {row['path']} (label: {row['label']})")
    print(f"Model input: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"min={tensor.min():.3f}, max={tensor.max():.3f}")
    print(f"Saved: {OUTPUTS_DIR / 'sample_grid.png'}")
    print(f"Saved: {OUTPUTS_DIR / 'preprocessing_example.png'}")


if __name__ == "__main__":
    main()