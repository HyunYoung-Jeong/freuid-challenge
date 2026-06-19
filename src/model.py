import timm
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=0,
    )
    feat_dim = model.num_features
    model = nn.Sequential(
        model,
        nn.Linear(feat_dim, 1),
    )
    return model
