# CLAUDE.md

notes for working in this repo. read this before touching code. written in the spirit of "the data is the thing and neural nets fail silently" — so most of this is about not fooling ourselves.

## what this is

the FREUID challenge 2026 (IJCAI-ECAI). binary problem: given an image of an identity document, predict whether it's **bona-fide** (genuine) or an **attack** (fraud). attacks come in three flavors — physical tampering, GenAI digital edits, and print-and-capture — across 7 under-represented document types (asian/african, latin + arabic scripts).

the whole game is **generalization**. the private test set will hit us with document types and attack styles we barely saw in training. a model that nails the training distribution and nothing else loses.

## the metric (memorize this)

- primary: **AuDET** — area under the DET curve. **lower is better.** not accuracy. not AUROC.
- operating point: **APCER @ 1% BPCER** — of all attacks, what fraction sneak through when we allow only 1% of genuine docs to be wrongly rejected. **lower is better.**
- consequence: the model outputs a **continuous attack score in [0,1]**, never a hard 0/1. the metric sweeps thresholds; a hard label throws that away.
- `src/metrics.py` must implement these exactly. local validation optimizes _this_ — not loss, not accuracy. if the local metric and the public leaderboard disagree by a lot, one of them is wrong: find out which before doing anything else.

## rules that bite

- code ships under an OSI-approved license at the end. keep `LICENSE` real and present.
- the technical report is a graded deliverable. `reports/report.md` gets updated as we go, not the night before.
- data is non-commercial research only. never commit it, never push it anywhere.
- public leaderboard is a small validation slice; final rank is a hidden test set. **do not tune against the public LB.** trust local validation.

## repo map

```
data/        raw/ (kaggle data, GITIGNORED), processed/, splits/
notebooks/   exploration only. disposable. logic does NOT live here.
src/         the actual code, importable.
  data.py            dataset, dataloaders, label parsing
  transforms.py      augmentations
  model.py           model defs
  train.py           training loop
  predict.py         inference -> attack scores
  metrics.py         AuDET + APCER@1%BPCER   <- get this right
  make_submission.py scores -> submission csv
  utils.py           seeding, logging, config loading
configs/     one yaml = one fully-specified experiment
outputs/     models/, predictions/, submissions/ — all GITIGNORED
reports/     report.md
tests/       test_metrics  <- sanity-checks the metric
```

## commands

```bash
# install
pip install -r requirements.txt

# train (a config fully describes a run)
python -m src.train --config configs/baseline.yaml

# inference on the test set -> attack scores
python -m src.predict --config configs/baseline.yaml

# build the submission csv
python -m src.make_submission --config configs/baseline.yaml

# tests (run before trusting the metric)
pytest tests/
```

## how we work here

- **logic in `src/`, never in notebooks.** notebooks are for looking at data and quick plots, then they get thrown away. anything worth keeping becomes a function in `src/`. also: we have to ship clean reproducible code, and notebook spaghetti doesn't qualify.
- **the config is the source of truth.** every run is fully described by a yaml — seed, paths, model, hyperparams, augments. report a score, point to the exact config that made it. no hidden state.
- **fix the seed.** everywhere, via `utils.set_seed()`. reproducibility isn't a nicety here, it's a grading requirement.
- **never commit data or checkpoints.** `data/` and `outputs/` are gitignored. about to `git add` something big? stop.
- **one change at a time, logged.** change five things and watch the score move, you've learned nothing.

## the recipe (do this in order)

1. **be one with the data.** before any modeling, look at images — genuine vs each attack type. hunt for corrupt files, duplicates, mislabels, weird aspect ratios, and the obvious shortcuts a model would cheat on. you should be able to describe this dataset out loud.
2. **skeleton + dumb baseline first.** get the full loop running end to end: load -> tiny model -> predict -> write a _valid_ submission -> score it locally -> submit once. the goal is not a good score. the goal is working plumbing and confidence it's correct.
3. **verify, because nets fail silently:**
   - loss at init should be about `-log(0.5) ≈ 0.69` for balanced binary cross-entropy. if it isn't, something's broken before you've trained anything.
   - overfit a single batch to ~zero loss. can't? the model/loss/data path is broken — no point training on the full set.
   - actually look at the tensors going _into_ the net (post-transform). that's where silent bugs hide: wrong normalization, flipped channels, an augmentation that nukes the forgery signal.
4. **then go bigger / regularize / tune** — only once the above is solid. don't be a hero on architecture; start from a known pretrained backbone and earn every change.

## gotchas specific to this comp

- **split for generalization.** random splits lie here. hold out whole document types and/or whole attack types in `data/splits/`, so local validation actually predicts the hidden-test pain. a glowing random-split number means little.
- **augmentation can erase the evidence.** the artifacts that reveal a forgery are often subtle — compression, print/scan noise, edge inconsistencies. aggressive augmentation can destroy exactly the signal we need. add augments deliberately and measure each one.
- **the domain gap is the point.** print-and-capture exists to kill detectors that lean on digital pixel noise. don't assume a clean digital-forgery detector transfers.
- **submission format must match `sample_submission` exactly** — same ids, same column, continuous scores. check it every time; a malformed file is a wasted submission slot.

## for claude (you, working in this repo)

- prefer the simplest thing that works. don't add a dependency, an abstraction, or a framework unless it's clearly earned.
- put new logic in `src/` as functions — not in notebooks, not inline in scripts.
- treat `src/metrics.py` as load-bearing. touch it, and add or keep a test in `tests/` against a hand-checked case.
- never write code that commits, uploads, or moves the dataset out of `data/`.
- when a change is more than a small edit (new module, refactor, new dependency, anything touching the metric or the split logic), say what you'll do and why _before_ doing it.
- unsure about the data layout? read it (`os.walk` over `/kaggle/input` or `data/raw`) instead of guessing filenames.
