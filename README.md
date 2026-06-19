# FREUID Challenge 2026 (IJCAI-ECAI) — Baseline Pipeline

Binary classification of identity document images: **bona fide** (genuine) vs **attack** (fraud). Attacks include physical tampering, GenAI digital edits, and print-and-capture, across 7 under-represented document types with Asian/African and Latin/Arabic scripts.

## Metric

| Metric | Description | Direction |
|---|---|---|
| **AuDET** (primary) | Area under the DET curve, computed as `1 - AuROC` | Lower is better |
| **APCER @ 1% BPCER** | Attack Presentation Classification Error Rate at 1% Bona fide Presentation Classification Error Rate | Lower is better |

The model outputs a continuous attack score in `[0, 1]` — never a hard label.

## Pipeline Overview

```
train_labels.csv ──> data.py ──> FREUIDDataset ──> train.py ──> model checkpoint
                        │                              │
                        │         transforms.py        │
                        │         (albumentations)     │
                        │                              │
                        └── val split by doc type      ├── sanity checks (init loss, batch overfit)
                                                       ├── per-epoch AuDET + APCER tracking
                                                       └── saves best model by val AuDET
                                                              │
                                                              v
                                              predict.py ──> scores CSV
                                                              │
                                                              v
                                              make_submission.py ──> submission CSV
```

## Model

ResNet-18 pretrained on ImageNet (via `timm`), with the classifier head replaced by a single `Linear(512, 1)` layer. Binary cross-entropy with logits loss. AdamW optimizer with cosine annealing schedule.

This is a deliberately simple baseline — the goal is validated plumbing, not a competitive score.

## Validation Strategy

Random train/val splits are misleading for this competition because the private test set contains unseen document types. Instead, we hold out an entire document type (`MAURITIUS/ID` by default) as the validation set. This simulates the domain gap the model will face at evaluation time.

Configured via `data.val_doc_type` in the YAML config.

## Transforms

Augmentation is intentionally conservative. Forgery artifacts (compression noise, edge inconsistencies, print/scan patterns) are subtle — aggressive augmentation can destroy the signal.

**Train:** Resize to 224x224, horizontal flip (p=0.5), mild color jitter (brightness/contrast/saturation 0.1, hue 0.05, p=0.3), ImageNet normalization.

**Validation/Test:** Resize to 224x224, ImageNet normalization only.

## Sanity Checks

The training loop runs two checks before training begins:

1. **Init loss check** — BCE loss at initialization should be ~`-log(0.5) = 0.693`. A large deviation signals a broken data/model path.
2. **Single-batch overfit** — 50 gradient steps on one batch should drive loss near zero. Failure means the model cannot learn from the data at all.

## Debug Mode

The config has a `debug` section to run on a small data subset locally:

```yaml
debug:
  enabled: true    # set to false for full training
  max_samples: 200 # samples per split (train, val, test)
```

When `enabled: true`, the pipeline limits train, val, and test sets to `max_samples` rows each. This lets you verify the full pipeline in minutes on CPU.

## Project Structure

```
configs/
  baseline.yaml        # fully specifies an experiment (seed, paths, model, hyperparams)
src/
  utils.py             # set_seed(), load_config(), parse_args()
  data.py              # FREUIDDataset, doc-type-based train/val split, dataloaders
  transforms.py        # train/val augmentations (albumentations)
  model.py             # build_model() — timm ResNet-18 + linear head
  metrics.py           # compute_audet(), compute_apcer_at_bpcer()
  train.py             # training loop with sanity checks and best-model saving
  predict.py           # inference on test set -> scores CSV
  make_submission.py   # scores CSV -> submission CSV
tests/
  test_metrics.py      # metric correctness tests (perfect, worst, random, ordering)
data/
  raw/                 # Kaggle data (gitignored)
outputs/
  models/              # checkpoints (gitignored)
  predictions/         # raw score CSVs (gitignored)
  submissions/         # final submission CSVs (gitignored)
```

## Usage

```bash
# install dependencies
pip install -r requirements.txt

# run metric tests
pytest tests/test_metrics.py -v

# train (config fully describes the run)
python -m src.train --config configs/baseline.yaml

# inference on test set
python -m src.predict --config configs/baseline.yaml

# generate submission CSV
python -m src.make_submission --config configs/baseline.yaml
```

## Dependencies

- PyTorch + torchvision
- timm (pretrained models)
- albumentations (image augmentation)
- scikit-learn (metric computation)
- pandas, numpy, Pillow, PyYAML, tqdm

## Data

Training data is not included in this repository. Place the Kaggle competition data under `data/raw/` with this layout:

```
data/raw/
  train_labels.csv
  train/train/*.jpeg          # 69,352 training images
  public_test/public_test/*.jpeg  # 7,821 test images
```

## License

See `LICENSE`.
