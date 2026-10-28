"""Phase 2, Step 4: train the v2 model (item classes -> 4 categories).

Run from the project root:
    python -m src.train_v2
    python -m src.train_v2 --head-epochs 1 --ft-epochs 0 --samples-per-epoch 600   (smoke test)

Only the 'train' and 'val' splits are loaded. The 'test' split is never read.
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
from sklearn.metrics import f1_score
from torchvision.transforms import v2

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED
from src.model_v2 import build_model_v2, load_model_v2
from src.preprocessing import load_image_bgr, resize_rgb
from src.taxonomy import CATEGORIES, ITEM_TO_CATEGORY, item_key
from src.v2_config import MODEL_V2_PATH, NEXTECH_DIR, OUTPUTS_V2, SPLIT_V2_CSV


class ImageSet(torch.utils.data.Dataset):
    """Preprocessed images in memory. On-the-fly augmentation only if train=True."""

    def __init__(self, rows, class_to_idx, train):
        self.images, self.labels = [], []
        for _, r in rows.iterrows():
            img = load_image_bgr(NEXTECH_DIR / r["path"])
            if img is None:
                raise RuntimeError(f"Cannot read {r['path']}")
            rgb = resize_rgb(img)  # same preprocessing as inference
            self.images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
            self.labels.append(class_to_idx[r["fine_class"]])
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


def category_matrix(classes):
    """(n_classes x 4) 0/1 matrix: which category each item class belongs to."""
    m = torch.zeros(len(classes), len(CATEGORIES))
    for i, c in enumerate(classes):
        m[i, CATEGORIES.index(ITEM_TO_CATEGORY[item_key(c)][0])] = 1.0
    return m


def set_frozen(model, k):
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= k


def set_train_mode(model, k):
    model.train()
    for block in list(model.features)[:k]:
        block.eval()


def train_epoch(model, loader, criterion, optimizer):
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, cat_m):
    model.eval()
    loss_sum, n, ys, probs = 0.0, 0, [], []
    for x, y in loader:
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        n += len(y)
        ys.append(y)
        probs.append(torch.softmax(out, dim=1))
    y = torch.cat(ys)
    p = torch.cat(probs)
    fine_to_cat = cat_m.argmax(1)
    y_cat = fine_to_cat[y].numpy()
    cat_pred = (p @ cat_m).argmax(1).numpy()
    fine_pred = p.argmax(1).numpy()
    b3 = CATEGORIES.index("b3")
    b3_mask = y_cat == b3
    return {
        "loss": loss_sum / n,
        "fine_acc": float((fine_pred == y.numpy()).mean()),
        "cat_acc": float((cat_pred == y_cat).mean()),
        "cat_f1": float(f1_score(y_cat, cat_pred, labels=list(range(len(CATEGORIES))),
                                 average="macro", zero_division=0)),
        "b3_recall": float((cat_pred[b3_mask] == b3).mean()) if b3_mask.any() else float("nan"),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--unfreeze-from", type=int, default=9)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--samples-per-epoch", type=int, default=2400,
                    help="images drawn (with replacement) per epoch by the balanced sampler")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(SPLIT_V2_CSV)
    df = df[df["split"].isin(["train", "val"])]  # test rows are dropped here
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    classes = sorted(train_df["fine_class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cat_m = category_matrix(classes)

    print(f"Train files: {len(train_df)} | Val files: {len(val_df)} (test split not loaded)")
    print(f"Item classes ({len(classes)}): {classes}")
    summary = pd.DataFrame({
        "category": [ITEM_TO_CATEGORY[item_key(c)][0] for c in classes],
        "train_files": [int((train_df['fine_class'] == c).sum()) for c in classes],
        "train_sources": [int(train_df.loc[train_df['fine_class'] == c, 'group_key'].nunique())
                          for c in classes],
        "val_files": [int((val_df['fine_class'] == c).sum()) for c in classes],
    }, index=classes)
    print(summary.to_string())
    missing_val = [c for c in classes if (val_df["fine_class"] == c).sum() == 0]
    if missing_val:
        print("WARNING: no val images for:", missing_val)

    print("\nLoading and preprocessing images into memory ...")
    train_set = ImageSet(train_df, class_to_idx, True)
    val_set = ImageSet(val_df, class_to_idx, False)

    # Balanced sampling: every class and every source image gets equal total weight.
    n_files_src = train_df.groupby("group_key")["path"].transform("size").to_numpy()
    n_src_cls = train_df.groupby("fine_class")["group_key"].transform("nunique").to_numpy()
    weights = 1.0 / (n_files_src * n_src_cls)
    gen = torch.Generator().manual_seed(SEED)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=args.samples_per_epoch, replacement=True, generator=gen)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()  # no class weights: the sampler balances classes

    print("Building model ...")
    model = build_model_v2(len(classes), pretrained=True)
    n_blocks = len(model.features)
    phases = [("head", args.head_epochs, n_blocks, args.lr_head)]
    if args.ft_epochs > 0:
        phases.append(("finetune", args.ft_epochs, args.unfreeze_from, args.lr_ft))

    history = []
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "state": None, "m": None}
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
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
            m = evaluate(model, val_loader, criterion, cat_m)
            history.append({"epoch": epoch_no, "phase": name, "train_loss": tr_loss,
                            "train_acc": tr_acc, **{f"val_{k}": v for k, v in m.items()}})
            improved = m["cat_f1"] > best["f1"] or (
                m["cat_f1"] == best["f1"] and m["loss"] < best["loss"])
            if improved:
                best.update(f1=m["cat_f1"], loss=m["loss"], epoch=epoch_no,
                            state=copy.deepcopy(model.state_dict()), m=m)
            print(f"epoch {epoch_no:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {m['loss']:.3f} item-acc {m['fine_acc']:.3f} "
                  f"cat-acc {m['cat_acc']:.3f} cat-F1 {m['cat_f1']:.3f} "
                  f"B3-recall {m['b3_recall']:.3f} | {time.time() - t0:.0f}s"
                  f"{'  <- best' if improved else ''}")

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time / 60:.1f} min")
    print(f"Best epoch: {best['epoch']} (val category macro-F1 {best['f1']:.4f})")
    print("Note: val is small and was used to pick this epoch, so these numbers are optimistic.")

    MODEL_V2_PATH.parent.mkdir(exist_ok=True)
    OUTPUTS_V2.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "classes": classes,
                "categories": CATEGORIES, "image_size": IMAGE_SIZE,
                "arch": "mobilenet_v3_small", "best_epoch": best["epoch"],
                "val_cat_f1": best["f1"]}, MODEL_V2_PATH)
    print(f"Saved model: {MODEL_V2_PATH} ({MODEL_V2_PATH.stat().st_size / 1e6:.1f} MB)")

    reloaded, _ = load_model_v2(MODEL_V2_PATH)
    re = evaluate(reloaded, val_loader, criterion, cat_m)
    ok = abs(re["cat_f1"] - best["f1"]) < 1e-3
    print(f"Reload check: val cat-F1 {re['cat_f1']:.4f} (expected {best['f1']:.4f}) -> "
          f"{'OK' if ok else 'MISMATCH'}")

    hist = pd.DataFrame(history)
    hist.to_csv(OUTPUTS_V2 / "training_history_v2.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train (sampled)")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["val_cat_acc"], label="val category accuracy")
    ax[1].plot(hist["epoch"], hist["val_cat_f1"], label="val category macro-F1")
    ax[1].plot(hist["epoch"], hist["val_fine_acc"], label="val item accuracy")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_V2 / "training_curves_v2.png", dpi=120)
    plt.close(fig)

    config = {
        "arch": "mobilenet_v3_small (torchvision, ImageNet pretrained)",
        "args": vars(args), "seed": SEED, "image_size": IMAGE_SIZE,
        "classes": classes, "categories": CATEGORIES,
        "class_to_category": {c: ITEM_TO_CATEGORY[item_key(c)][0] for c in classes},
        "train_files": len(train_df), "val_files": len(val_df),
        "per_class": summary.reset_index(names="class").to_dict(orient="records"),
        "best_epoch": best["epoch"], "best_val_metrics": best["m"],
        "training_time_seconds": round(total_time, 1),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": "cpu",
    }
    (OUTPUTS_V2 / "training_config_v2.json").write_text(json.dumps(config, indent=2))
    print("Saved: outputs\\v2\\training_history_v2.csv, training_curves_v2.png, "
          "training_config_v2.json")


if __name__ == "__main__":
    main()