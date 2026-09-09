"""Phase A / Stage 1: coarse training at 640 on yolov8s-p2 (CLAHE grayscale data).

Session-timeout safe: checkpoints every epoch; if last.pt exists under
--project, the run auto-resumes instead of restarting. Point --project at a
persistent location (Google Drive on Colab; /kaggle/working on Kaggle, then
save the notebook version to keep outputs).

    python train_stage1.py --data /kaggle/input/<clahe-ds>/data.yaml \
        --project /kaggle/working/runs_phaseA

Notes vs. the raw plan:
- cache=True is OFF: Kaggle input dirs are read-only and Colab RAM can't hold
  115k images. workers=4 JPEG-decoding grayscale is fast enough.
- yolov8s-p2 has no official .pt, so we build from yaml and load yolov8s.pt
  backbone weights (transfer learning instead of from-scratch).
- batch=-1 = AutoBatch (~60% VRAM). Set a fixed batch if using 2 GPUs (DDP
  does not support AutoBatch): --batch 32 --device 0,1
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RUN_NAME = "stage1_640_coarse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default="runs_phaseA")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", default=None,
                    help="warm-start weights (e.g. previous best.pt); "
                         "backbone transfers, head resets on nc mismatch")
    ap.add_argument("--copy-paste", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    last = Path(args.project) / RUN_NAME / "weights" / "last.pt"
    if last.exists():
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        return

    if args.init:
        print(f"Warm start from {args.init} (nc mismatch -> backbone transfer, fresh head)")
        model = YOLO(args.init)
    else:
        model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=10,
        copy_paste=args.copy_paste,
        # Orientation is class-defining (npn vs pnp, nmos vs pmos): no flips.
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        # Grayscale input: hue/sat jitter is meaningless, keep brightness only.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        project=args.project,
        name=RUN_NAME,
        exist_ok=True,
        save_period=1,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
