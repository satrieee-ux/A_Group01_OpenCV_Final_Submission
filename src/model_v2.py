"""v2 model: MobileNetV3-Small with a configurable number of output classes."""
import torch
import torch.nn as nn
from torchvision import models


def build_model_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_model_v2(path):
    """Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model_v2(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt