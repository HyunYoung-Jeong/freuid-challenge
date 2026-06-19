import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class FREUIDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = "label" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpeg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        sample = {"image": image, "id": row["id"]}

        if self.has_labels:
            sample["label"] = torch.tensor(row["label"], dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Overlay dataset — face-region crop with cache
# ---------------------------------------------------------------------------

_mtcnn_instance = None


def _get_mtcnn():
    global _mtcnn_instance
    if _mtcnn_instance is None:
        from facenet_pytorch import MTCNN
        _mtcnn_instance = MTCNN(keep_all=False, device="cpu", post_process=False)
    return _mtcnn_instance


class OverlayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform, cfg: dict):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = "label" in df.columns

        self.crop_margin = cfg["data"].get("crop_margin", 0.75)
        self.cache_dir = cfg["data"].get("crop_cache_dir", "data/processed/overlay_crops")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.fallback_count = 0

    def __len__(self):
        return len(self.df)

    def _get_crop(self, img_path: str, image_id: str) -> np.ndarray:
        cache_path = os.path.join(self.cache_dir, f"{image_id}.png")
        if os.path.exists(cache_path):
            img = cv2.imread(cache_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        pil_img = Image.fromarray(image)
        mtcnn = _get_mtcnn()
        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            bw, bh = x2 - x1, y2 - y1
            mx = bw * self.crop_margin
            my = bh * self.crop_margin
            x1 = max(0, int(x1 - mx))
            y1 = max(0, int(y1 - my))
            x2 = min(w, int(x2 + mx))
            y2 = min(h, int(y2 + my))
            crop = image[y1:y2, x1:x2]
        else:
            self.fallback_count += 1
            side = int(min(h, w) * 0.6)
            cy, cx = h // 2, w // 2
            y1 = cy - side // 2
            x1 = cx - side // 2
            crop = image[y1:y1 + side, x1:x1 + side]

        cv2.imwrite(cache_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpeg")
        image = self._get_crop(img_path, row["id"])

        if self.transform:
            image = self.transform(image=image)["image"]

        sample = {"image": image, "id": row["id"]}
        if self.has_labels:
            sample["label"] = torch.tensor(row["label"], dtype=torch.float32)
        return sample


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _maybe_truncate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    debug = cfg.get("debug", {})
    if debug.get("enabled", False):
        n = debug.get("max_samples", 200)
        return df.head(n)
    return df


def get_train_val_split(csv_path: str, val_doc_type: str):
    df = pd.read_csv(csv_path)
    train_df = df[df["type"] != val_doc_type].copy()
    val_df = df[df["type"] == val_doc_type].copy()
    return train_df, val_df


def build_dataloaders(cfg: dict, train_transform, val_transform):
    train_df, val_df = get_train_val_split(
        cfg["data"]["train_csv"], cfg["data"]["val_doc_type"]
    )
    train_df = _maybe_truncate(train_df, cfg)
    val_df = _maybe_truncate(val_df, cfg)
    img_dir = cfg["data"]["train_img_dir"]

    train_ds = FREUIDDataset(train_df, img_dir, transform=train_transform)
    val_ds = FREUIDDataset(val_df, img_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_overlay_dataloaders(cfg: dict, train_transform, val_transform):
    train_df, val_df = get_train_val_split(
        cfg["data"]["train_csv"], cfg["data"]["val_doc_type"]
    )
    train_df = _maybe_truncate(train_df, cfg)
    val_df = _maybe_truncate(val_df, cfg)
    img_dir = cfg["data"]["train_img_dir"]

    train_ds = OverlayDataset(train_df, img_dir, train_transform, cfg)
    val_ds = OverlayDataset(val_df, img_dir, val_transform, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(
        "overlay dataloaders: train=%d, val=%d", len(train_ds), len(val_ds)
    )
    return train_loader, val_loader


def build_test_dataloader(cfg: dict, transform):
    test_dir = cfg["data"]["test_img_dir"]
    ids = [f.replace(".jpeg", "") for f in os.listdir(test_dir) if f.endswith(".jpeg")]
    test_df = pd.DataFrame({"id": ids})
    test_df = _maybe_truncate(test_df, cfg)
    test_ds = FREUIDDataset(test_df, test_dir, transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return test_loader


def build_overlay_test_dataloader(cfg: dict, transform):
    test_dir = cfg["data"]["test_img_dir"]
    ids = [f.replace(".jpeg", "") for f in os.listdir(test_dir) if f.endswith(".jpeg")]
    test_df = pd.DataFrame({"id": ids})
    test_df = _maybe_truncate(test_df, cfg)
    test_ds = OverlayDataset(test_df, test_dir, transform, cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader
