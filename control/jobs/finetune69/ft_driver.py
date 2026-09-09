"""Finetune driver (self-contained single file): attach ready dataset + old
weights, transfer backbone (92->69 auto), train.

Inputs (attached datasets):
- circuitvision-focused69 : extracted tree, data.yaml nc=69  -> /kaggle/input/...
- yolo-checkpoints        : best_92cls.pt (100-120 epoch stage-1 backbone)

Transfer is proven: ultralytics prints "Overriding nc=92 with nc=69" and
"Transferred 389/437 items" (backbone kept, head reset). Nothing wasted.
Resume: seeds last.pt from kernel_sources (chain) or restarts from old weights.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def log(msg):
    print(msg, flush=True)


def ensure_ultralytics():
    try:
        import ultralytics  # noqa
        log(f"ultralytics {ultralytics.__version__} present")
        return
    except ImportError:
        pass
    log("installing ultralytics ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    assert r.returncode == 0, "pip install ultralytics failed"
    import ultralytics  # noqa
    log(f"ultralytics {ultralytics.__version__} installed")


def find_ready_root():
    roots = [p for p in INPUT.rglob("*")
             if p.is_dir() and (p / "train" / "images").is_dir()
             and (p / "data.yaml").is_file()]
    assert roots, "no dataset root with data.yaml under /kaggle/input"
    pref = [p for p in roots if "focused69" in str(p)]
    root = pref[0] if pref else roots[0]
    log(f"using data root: {root}")
    return root


def find_weights():
    cands = sorted(INPUT.rglob("best_92cls.pt"))
    assert cands, "best_92cls.pt not attached (need yolo-checkpoints dataset)"
    log(f"using init weights: {cands[-1]}")
    return cands[-1]


def find_seed(project):
    cands = sorted(INPUT.rglob("stage1_640_coarse/weights/last.pt"))
    ft = list(project.rglob("weights/last.pt"))
    if ft:
        return None  # local resume handled by ultralytics directly
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--project", default="runs_ft69")
    ap.add_argument("--name", default="ft69_640")
    args = ap.parse_args()

    ensure_ultralytics()
    import torch
    from ultralytics import YOLO

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"visible GPUs: {n_gpu}")
    assert n_gpu >= 1, "no GPU visible — refusing CPU burn"
    device = args.device or ("0,1" if n_gpu >= 2 else "0")
    batch = args.batch if args.batch is not None else (32 if n_gpu >= 2 else 16)
    torch.zeros(1, device="cuda")
    log(f"cuda smoke OK; device={device} batch={batch}")

    root = find_ready_root()
    data_yaml = root / "data.yaml"
    project = WORK / args.project

    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        log(f"resuming local {last}")
        YOLO(str(last)).train(resume=True)
        return
    seed = find_seed(project)
    if seed is not None:
        dest = project / args.name / "weights" / "last.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(seed, dest)
        log(f"seeded {seed}, resuming")
        YOLO(str(dest)).train(resume=True)
        return

    w0 = find_weights()
    log(f"finetuning from 92-class backbone {w0} -> 69-class head (fresh head, kept backbone)")
    model = YOLO(str(w0))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=batch,
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=5,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        project=str(project),
        name=args.name,
        exist_ok=True,
        save_period=1,
        patience=15,
    )
    log("finetune done")


if __name__ == "__main__":
    main()
