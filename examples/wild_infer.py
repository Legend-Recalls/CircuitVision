import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
"""Spot-check inference on demo/inputs. Saves annotated images to demo/outputs."""
from pathlib import Path

from ultralytics import YOLO

HERE = _ROOT
WEIGHTS = HERE / "control" / "runs" / "round3_68_v1" / "best.pt"

m = YOLO(str(WEIGHTS))
results = m.predict(
    source=str(HERE / "demo" / "inputs"),
    conf=0.25, device="cpu", verbose=False,
    project=str(HERE / "demo" / "outputs"),
    name="pred", exist_ok=True, save=True,
)
for x in results:
    print("===", x.path)
    if x.boxes is None or len(x.boxes) == 0:
        print("  no detections")
        continue
    print("  saved:", x.save_dir)
    dets = sorted(zip(x.boxes.cls.tolist(), x.boxes.conf.tolist()),
                  key=lambda t: -t[1])
    for cls, conf in dets:
        print(f"  {x.names[int(cls)]} {conf:.2f}")
