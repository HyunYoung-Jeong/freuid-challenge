import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
