import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import build_test_dataloader
from src.model import build_model
from src.transforms import get_val_transforms
from src.utils import load_config, parse_args, set_seed


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_ids = []
    all_scores = []
    for batch in tqdm(loader, desc="predict"):
        images = batch["image"].to(device)
        logits = model(images).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_ids.extend(batch["id"])
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    return all_ids, all_scores


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt_path = os.path.join(
        cfg["output"]["model_dir"], f"{cfg['output']['experiment_name']}_best.pt"
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"loaded checkpoint: {ckpt_path}")

    img_size = cfg["data"]["img_size"]
    test_loader = build_test_dataloader(cfg, get_val_transforms(img_size))

    ids, scores = predict(model, test_loader, device)

    pred_dir = cfg["output"]["pred_dir"]
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, f"{cfg['output']['experiment_name']}.csv")
    pd.DataFrame({"id": ids, "score": scores}).to_csv(out_path, index=False)
    print(f"predictions saved: {out_path} ({len(ids)} samples)")


if __name__ == "__main__":
    main()
