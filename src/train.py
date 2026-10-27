"""Stage 4: train the classifier (transfer learning).

Run from the project root:
    python -m src.train                                  (full training)
    python -m src.train --head-epochs 1 --ft-epochs 0    (quick smoke test)

Only the 'train' and 'val' splits are used. The 'test' split is never loaded.
"""
import argparse
import copy
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH, MODELS_DIR, OUTPUTS_DIR,
    PROJECT_ROOT, SEED, SPLIT_CSV, TARGET_CLASSES,
)
from src.model import build_model, load_model
from src.preprocessing import load_image_bgr, resize_rgb

CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}


class ImageSet(torch.utils.data.Dataset):
    """Holds preprocessed images in memory. Augmentation only if train=True."""

    def __init__(self, rows: pd.DataFrame, train: bool):
        self.images, self.labels = [], []
        for _, r in rows.iterrows():
            img = load_image_bgr(PROJECT_ROOT / r["path"])
            if img is None:
                raise RuntimeError(f"Cannot read {r['path']}")
            rgb = resize_rgb(img)  # same preprocessing as inference
            self.images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
            self.labels.append(CLASS_TO_IDX[r["label"]])
        self.augment = None
        if train:
            self.augment = v2.Compose([
                v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.augment is not None:
            x = self.augment(x)
        x = (x.float() / 255.0 - self.mean) / self.std
        return x, self.labels[i]


def set_frozen(model, k):
    """Freeze feature blocks [0, k); blocks >= k and the classifier are trainable."""
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= k


def set_train_mode(model, k):
    model.train()
    for block in list(model.features)[:k]:
        block.eval()  # keep BatchNorm statistics of frozen blocks fixed


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--unfreeze-from", type=int, default=9,
                    help="fine-tune feature blocks with index >= this value")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(SPLIT_CSV)
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    print(f"Train images: {len(train_df)} | Val images: {len(val_df)} (test split not loaded)")

    print("Loading and preprocessing images into memory ...")
    train_set, val_set = ImageSet(train_df, True), ImageSet(val_df, False)
    gen = torch.Generator().manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    counts = np.bincount(train_set.labels, minlength=len(TARGET_CLASSES))
    class_weights = counts.sum() / (len(counts) * counts)
    print("Train images per class:", dict(zip(TARGET_CLASSES, counts.tolist())))
    print("Class weights:", dict(zip(TARGET_CLASSES, class_weights.round(3).tolist())))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    print("Building model (downloads pretrained weights on first run) ...")
    model = build_model(pretrained=True)
    n_blocks = len(model.features)

    phases = [("head", args.head_epochs, n_blocks, args.lr_head)]
    if args.ft_epochs > 0:
        phases.append(("finetune", args.ft_epochs, args.unfreeze_from, args.lr_ft))

    history, best = [], {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    start, epoch_no = time.time(), 0

    for name, n_epochs, frozen_k, lr in phases:
        set_frozen(model, frozen_k)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        print(f"\n== Phase '{name}': {n_epochs} epochs, lr={lr}, "
              f"trainable params={sum(p.numel() for p in trainable):,}")
        for _ in range(n_epochs):
            epoch_no += 1
            t0 = time.time()
            set_train_mode(model, frozen_k)
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
            model.eval()
            va_loss, va_acc = run_epoch(model, val_loader, criterion)
            history.append({"epoch": epoch_no, "phase": name, "train_loss": tr_loss,
                            "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
            improved = va_acc > best["acc"] or (va_acc == best["acc"] and va_loss < best["loss"])
            if improved:
                best.update(acc=va_acc, loss=va_loss, epoch=epoch_no,
                            state=copy.deepcopy(model.state_dict()))
            print(f"epoch {epoch_no:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - t0:.0f}s"
                  f"{'  <- best' if improved else ''}")

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time / 60:.1f} min")
    print(f"Best epoch: {best['epoch']} (val acc {best['acc']:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best["state"], "classes": list(TARGET_CLASSES),
                "image_size": IMAGE_SIZE, "arch": "mobilenet_v3_small",
                "best_epoch": best["epoch"], "val_acc": best["acc"]}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # Verify that the saved model reloads and reproduces the best validation accuracy.
    reloaded, _ = load_model(MODEL_PATH)
    _, re_acc = run_epoch(reloaded, val_loader, criterion)
    print(f"Reload check: val acc {re_acc:.4f} (expected {best['acc']:.4f}) -> "
          f"{'OK' if abs(re_acc - best['acc']) < 1e-3 else 'MISMATCH'}")

    hist = pd.DataFrame(history)
    hist.to_csv(OUTPUTS_DIR / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train")
    ax[1].plot(hist["epoch"], hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "training_curves.png", dpi=120)
    plt.close(fig)

    config = {
        "arch": "mobilenet_v3_small (torchvision, ImageNet pretrained)",
        "args": vars(args), "seed": SEED, "image_size": IMAGE_SIZE,
        "train_images": len(train_df), "val_images": len(val_df),
        "train_per_class": dict(zip(TARGET_CLASSES, counts.tolist())),
        "best_epoch": best["epoch"], "best_val_acc": best["acc"],
        "training_time_seconds": round(total_time, 1),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": "cpu",
    }
    (OUTPUTS_DIR / "training_config.json").write_text(json.dumps(config, indent=2))
    print("Saved: outputs\\training_history.csv, training_curves.png, training_config.json")


if __name__ == "__main__":
    main()