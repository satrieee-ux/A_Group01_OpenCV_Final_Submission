"""Model definition and loading, shared by training, evaluation and the app."""
import torch
import torch.nn as nn
from torchvision import models

from src.config import TARGET_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with the final layer replaced by a 3-class layer."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(TARGET_CLASSES))
    return model


def load_model(path):
    """Load a trained checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if list(ckpt["classes"]) != list(TARGET_CLASSES):
        raise ValueError(f"Checkpoint classes {ckpt['classes']} != {TARGET_CLASSES}")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt