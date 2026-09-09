"""Stage-2 driver (self-contained): 1024px localization finetune, frozen backbone.

Inputs (attached):
- circuitvision-focused69 : ready 69-class tree + data.yaml
- yolo-checkpoints        : best_focused69_s1.pt (stage-1 weights)

Recipe mirrors train_stage2.py: freeze first 10 layers (edge detectors),
lr0 1e-4, close_mosaic 5, patience 8. Batch defaults low: 1024px + P2 head
is VRAM-hungry (batch 8 across 2x16GB; override with --batch if OOM/slack).
Resume: local last.pt, else seed from kernel_sources, else fresh weights.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUN_NAME = "stage2_1024_finetune"


def log(msg):
    print(msg, flush=True)


def ensure_ultralytics():
    try:
        import ultralytics  # noqa
        log("ultralytics present")
        return
    except ImportError:
        pass
    log("installing ultralytics ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    assert r.returncode == 0, "pip install failed"
    log("ultralytics installed")


def find_ready_root():
    roots = [p for p in INPUT.rglob("*")
             if p.is_dir() and (p / "train" / "images").is_dir()
             and (p / "data.yaml").is_file()]
    assert roots, "no dataset root with data.yaml"
    pref = [p for p in roots if "focused69" in str(p)]
    root = pref[0] if pref else roots[0]
    log(f"using data root: {root}")
    return root


def find_init_weights():
    cands = sorted(INPUT.rglob("best_focused69_s1.pt"))
    assert cands, "best_focused69_s1.pt not attached (need yolo-checkpoints)"
    log(f"using init weights: {cands[-1]}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs_stage2")
    args = ap.parse_args()

    ensure_ultralytics()
    import torch
    from ultralytics import YOLO

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"visible GPUs: {n_gpu}")
    assert n_gpu >= 1, "no GPU visible"
    device = args.device or ("0,1" if n_gpu >= 2 else "0")
    torch.zeros(1, device="cuda")
    log(f"cuda smoke OK; device={device} batch={args.batch}")

    root = find_ready_root()
    project = WORK / args.project

    last = project / RUN_NAME / "weights" / "last.pt"
    if last.exists():
        log(f"resuming local {last}")
        YOLO(str(last)).train(resume=True)
        return
    seeds = sorted(INPUT.rglob(f"{RUN_NAME}/weights/last.pt"))
    local = list(project.rglob("weights/last.pt"))
    if seeds and not local:
        dest = project / RUN_NAME / "weights" / "last.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(seeds[-1], dest)
        log(f"seeded {seeds[-1]}, resuming")
        YOLO(str(dest)).train(resume=True)
        return

    w0 = find_init_weights()
    log("stage-2 finetune: freeze=10, lr0=1e-4, 1024px")
    YOLO(str(w0)).train(
        data=str(root / "data.yaml"),
        epochs=args.epochs,
        imgsz=1024,
        batch=args.batch,
        device=device,
        workers=4,
        lr0=0.0001,
        cos_lr=True,
        freeze=10,
        close_mosaic=5,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        project=str(project),
        name=RUN_NAME,
        exist_ok=True,
        save_period=1,
        patience=8,
    )
    log("stage-2 done")


if __name__ == "__main__":
    main()
