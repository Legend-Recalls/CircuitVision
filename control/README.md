# control — CircuitVision Kaggle control plane (lives inside circuitvision/)

Batch control for Kaggle training: push script kernels, poll status,
download outputs. No browser needed. Token is **env-only, never stored**.

## Setup

```powershell
python -m pip install -r requirements.txt
$env:KAGGLE_API_TOKEN = 'KGAT_xxx'   # kaggle.com/settings -> Create New Token
```

Never commit a token. `.env` (if you use one) is git-ignored.

## Usage

```powershell
python kaggle_ctl.py kernels                 # your notebooks
python kaggle_ctl.py datasets                # your datasets
python kaggle_ctl.py status <user/slug>      # run status
python kaggle_ctl.py push jobs/hist          # push folder as run (save & run)
python kaggle_ctl.py output <user/slug> out/ # download version outputs
python kaggle_ctl.py pull <user/slug> tmp/   # download notebook source
```

## Layout

- `kaggle_ctl.py` — CLI: kernels/datasets/status/push/output/pull
- `jobs/hist/` — CPU histogram job (class counts for cut decisions)
- `jobs/train_stage1/` — GPU job: copy dataset -> remap -> train (resume-aware)
- `runs/` — downloaded outputs (git-ignored)
