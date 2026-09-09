"""Diagnose pin placement on vertical (h>w) boxes: print + zoomed crops."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
from circuitvision.graph_schema import dedup_components
from circuitvision.pin_templates import pins_for

WEIGHTS = _ROOT / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = _ROOT / "demo" / "inputs" / "smps.jpg"
OUTDIR = _ROOT / "demo" / "outputs" / "vert_debug"
CONF = 0.35


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=CONF, device="cpu", verbose=False)[0]
    raw = [(res.names[int(b.cls)], float(b.conf),
            tuple(float(v) for v in b.xyxy[0].tolist())) for b in res.boxes]
    dets = dedup_components(raw, 0.5)
    im = Image.open(SRC).convert("RGB")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls, conf, bbox in dets:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if h <= w:
            continue
        orient = 90
        pins = pins_for(cls, bbox, orient)
        print(f"{cls:35s} conf={conf:.2f} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}) "
              f"aspect={h / w:.1f} pins={[(r, round(x), round(y)) for r, x, y, _ in pins]}")
        if n < 6:
            pad = 40
            crop = im.crop((max(0, int(x0) - pad), max(0, int(y0) - pad),
                            int(x1) + pad, int(y1) + pad))
            s = max(1, 500 // max(crop.size))
            crop = crop.resize((crop.size[0] * s, crop.size[1] * s), Image.NEAREST)
            d = ImageDraw.Draw(crop)
            ox, oy = max(0, int(x0) - pad), max(0, int(y0) - pad)
            for r, px, py, _ in pins:
                X, Y = (px - ox) * s, (py - oy) * s
                d.ellipse([X - 8, Y - 8, X + 8, Y + 8], fill="red", outline="yellow")
                d.text((X + 10, Y - 10), r, fill="red")
            crop.save(OUTDIR / f"vert_{n}_{cls}.png")
            n += 1
    print("crops:", n)


if __name__ == "__main__":
    main()
