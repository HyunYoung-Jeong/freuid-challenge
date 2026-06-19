import timm
import torch
import torch.nn as nn

from src.transforms import BayarConv2d, SRMConv2d, IMAGENET_MEAN, IMAGENET_STD


class NoiseStream(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        frontend_type = cfg["model"].get("noise_frontend", "bayar")
        if frontend_type == "srm":
            self.frontend = SRMConv2d(in_channels=3)
            fe_out = self.frontend.out_channels
        else:
            self.frontend = BayarConv2d(3, 16, kernel_size=5)
            fe_out = 16

        feat_dim = cfg["model"].get("noise_feat_dim", 128)
        self.body = nn.Sequential(
            nn.Conv2d(fe_out, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.body(self.frontend(x))


class TwoStreamOverlayNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.noise_stream = NoiseStream(cfg)
        total_feat = self.noise_stream.feat_dim

        self.use_rgb = cfg["model"].get("use_rgb_stream", False)
        if self.use_rgb:
            self.rgb_backbone = timm.create_model(
                cfg["model"]["rgb_backbone"],
                pretrained=cfg["model"].get("rgb_pretrained", True),
                num_classes=0,
            )
            self.register_buffer(
                "rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            )
            total_feat += self.rgb_backbone.num_features

        fusion_dim = cfg["model"].get("fusion_dim", 128)
        self.head = nn.Sequential(
            nn.Linear(total_feat, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x):
        noise_feat = self.noise_stream(x)
        if self.use_rgb:
            x_norm = (x - self.rgb_mean) / self.rgb_std
            rgb_feat = self.rgb_backbone(x_norm)
            fused = torch.cat([noise_feat, rgb_feat], dim=1)
        else:
            fused = noise_feat
        return self.head(fused)


def build_model(cfg: dict) -> nn.Module:
    if cfg["model"].get("type") == "overlay":
        return TwoStreamOverlayNet(cfg)

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
