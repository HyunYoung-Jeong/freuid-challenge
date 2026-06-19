import os

import pandas as pd

from src.utils import load_config, parse_args


def main():
    args = parse_args()
    cfg = load_config(args.config)

    pred_path = os.path.join(
        cfg["output"]["pred_dir"], f"{cfg['output']['experiment_name']}.csv"
    )
    preds = pd.read_csv(pred_path)

    sub_dir = cfg["output"]["sub_dir"]
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, f"{cfg['output']['experiment_name']}.csv")

    submission = preds[["id", "score"]].copy()
    submission.columns = ["id", "score"]
    submission["score"] = submission["score"].clip(0.0, 1.0)
    submission.to_csv(sub_path, index=False)
    print(f"submission saved: {sub_path} ({len(submission)} rows)")
    print(f"score range: [{submission['score'].min():.4f}, {submission['score'].max():.4f}]")


if __name__ == "__main__":
    main()
