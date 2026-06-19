import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import build_dataloaders, build_overlay_dataloaders
from src.metrics import compute_apcer_at_bpcer, compute_audet
from src.model import build_model
from src.transforms import (
    get_train_transforms, get_val_transforms,
    get_overlay_train_transforms, get_overlay_val_transforms,
)
from src.utils import load_config, parse_args, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_labels = []
    all_scores = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.cpu().numpy())

        total_loss += loss.item() * len(labels)
        n += len(labels)

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    avg_loss = total_loss / n
    audet = compute_audet(all_labels, all_scores)
    apcer = compute_apcer_at_bpcer(all_labels, all_scores)
    return avg_loss, audet, apcer


def sanity_check_init_loss(model, loader, criterion, device):
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    with torch.no_grad():
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels).item()
    expected = -np.log(0.5)
    print(f"[sanity] init loss: {loss:.4f} (expected ~{expected:.4f})")
    if abs(loss - expected) > 0.3:
        print("[sanity] WARNING: init loss is far from expected -- check data/model")


def sanity_check_overfit_batch(model, loader, criterion, optimizer, device):
    model.train()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    print("[sanity] overfitting single batch...")
    for i in range(50):
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"  step {i+1}: loss={loss.item():.4f}")
    if loss.item() > 0.1:
        print("[sanity] WARNING: could not overfit batch -- check model/data path")
    else:
        print("[sanity] batch overfit OK")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    img_size = cfg["data"]["img_size"]
    is_overlay = cfg["model"].get("type") == "overlay"
    if is_overlay:
        train_tf = get_overlay_train_transforms(img_size)
        val_tf = get_overlay_val_transforms(img_size)
        train_loader, val_loader = build_overlay_dataloaders(cfg, train_tf, val_tf)
    else:
        train_tf = get_train_transforms(img_size)
        val_tf = get_val_transforms(img_size)
        train_loader, val_loader = build_dataloaders(cfg, train_tf, val_tf)
    print(f"train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    sanity_check_init_loss(model, train_loader, criterion, device)
    sanity_check_overfit_batch(model, train_loader, criterion, optimizer, device)

    # re-init model after sanity checks
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    model_dir = cfg["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    best_audet = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_audet, val_apcer = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"epoch {epoch+1}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_audet={val_audet:.4f} | val_apcer@1%={val_apcer:.4f}"
        )

        if val_audet < best_audet:
            best_audet = val_audet
            ckpt_path = os.path.join(
                model_dir, f"{cfg['output']['experiment_name']}_best.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model (audet={best_audet:.4f})")

    print(f"training done. best val_audet={best_audet:.4f}")


if __name__ == "__main__":
    main()
