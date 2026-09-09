"""Independent pin-on-wire measurement: v2 policy vs v3 policy, same boxes.

v2 proxy (what the old demo produced): aspect orientation + raw templates,
no snapping. v3: gate 1.5/0.1 + dark snapping + IC stubs.
On-wire = any dark pixel within 5px (same radius as _disc_score).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_schema import dedup_components
from pin_templates import pins_for, score_orientation, ic_stub_pins, ICS

WEIGHTS = Path(__file__).resolve().parent / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = Path(__file__).resolve().parent / "demo" / "inputs" / "smps.jpg"


def on_wire(dark, px, py, r=5):
    h, w = dark.shape
    x0, x1 = max(0, int(px) - r), min(w, int(px) + r + 1)
    y0, y1 = max(0, int(py) - r), min(h, int(py) + r + 1)
    return bool(dark[y0:y1, x0:x1].any())


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=0.35, device="cpu", verbose=False)[0]
    raw = [(res.names[int(b.cls)], float(b.conf),
            tuple(float(v) for v in b.xyxy[0].tolist())) for b in res.boxes]
    dets = dedup_components(raw, 0.5)
    dark = np.array(Image.open(SRC).convert("L")) < 128

    b_hit = b_tot = 0
    for cls, conf, bbox in dets:
        x0, y0, x1, y1 = bbox
        base = 0 if (x1 - x0) >= (y1 - y0) else 90
        for _, px, py, _ in pins_for(cls, bbox, base):
            b_tot += 1
            b_hit += on_wire(dark, px, py)

    a_hit = a_tot = n_flip = n_stub = 0
    for cls, conf, bbox in dets:
        x0, y0, x1, y1 = bbox
        base = 0 if (x1 - x0) >= (y1 - y0) else 90
        orient, score, margin = score_orientation(cls, bbox, dark)
        if not (margin >= 1.5 and score > 0.1):
            orient = base
        if orient != base:
            n_flip += 1
        made = []
        if cls in ICS:
            stubs = ic_stub_pins(bbox, dark)
            if stubs:
                made = [(r, x, y) for r, x, y, _ in stubs]
                n_stub += 1
        if not made:
            made = [(r, x, y) for r, x, y, _ in pins_for(cls, bbox, orient, dark=dark)]
        for _, px, py in made:
            a_tot += 1
            a_hit += on_wire(dark, px, py)

    print(f"BEFORE (v2 policy): {b_hit}/{b_tot} = {b_hit / b_tot:.2%}")
    print(f"AFTER  (v3 policy): {a_hit}/{a_tot} = {a_hit / a_tot:.2%}")
    print(f"flips: {n_flip}, stub ICs: {n_stub}")


if __name__ == "__main__":
    main()
