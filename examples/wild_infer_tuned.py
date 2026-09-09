import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
"""Tuned inference for dense sheets (see wild_infer.py).

Cross-class NMS kills stacked duplicates (resistor+capacitor on one
symbol); tighter iou + higher conf cut the tail. 127 -> 111 on smps.
"""
from pathlib import Path

from ultralytics import YOLO

HERE = _ROOT
WEIGHTS = HERE / "control" / "runs" / "round3_68_v1" / "best.pt"

m = YOLO(str(WEIGHTS))
results = m.predict(
    source=str(HERE / "demo" / "inputs" / "smps.jpg"),
    conf=0.40, iou=0.5, agnostic_nms=True, device="cpu", verbose=False,
    project=str(HERE / "demo" / "outputs"),
    name="pred_tuned", exist_ok=True, save=True,
)
for x in results:
    n = 0 if x.boxes is None else len(x.boxes)
    print("detections:", n)
