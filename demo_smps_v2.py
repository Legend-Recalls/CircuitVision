"""Demo v2: detect -> dedup -> orient -> pins (templates + IC stubs).

Run from D:/migration/new/circuitvision. CPU only.
Fixes vs v1: IoU dedup kills duplicate pin clusters; wire-pixel scoring
fixes transistor orientation + axial flips; IC stubs replace pitch guesses
where real wire entries are visible.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_schema import Component, Pin, CircuitGraph, dedup_components
from pin_templates import pins_for, score_orientation, ic_stub_pins, ICS

WEIGHTS = Path(__file__).resolve().parent / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = Path(__file__).resolve().parent / "demo" / "inputs" / "smps.jpg"
OUT = Path(__file__).resolve().parent / "demo" / "outputs" / "smps_pins_v2.jpg"
CONF = 0.35


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=CONF, device="cpu", verbose=False)[0]
    raw = [(res.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist()))
           for b in res.boxes]
    dets = dedup_components(raw, 0.5)
    print(f"dets: {len(raw)} -> {len(dets)} after IoU dedup")

    gray = Image.open(SRC).convert("L")
    dark = np.array(gray) < 128

    g = CircuitGraph()
    n_flip, n_stub = 0, 0
    for i, (cls, conf, bbox) in enumerate(dets):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        base = 0 if w >= h else 90
        orient, score, margin = score_orientation(cls, bbox, dark)
        # Flip only on decisive margins: sparse wire maps make near-ties
        # random. Defaults (aspect) stand otherwise.
        if not (margin >= 1.5 and score > 0.1):
            orient = base
        if orient != base:
            n_flip += 1
        g.components.append(Component(i, cls, conf, bbox, orient))
        made = []
        if cls in ICS:
            stubs = ic_stub_pins(bbox, dark)
            if stubs:
                made = [(r, x, y, c, "stub") for r, x, y, c in stubs]
                n_stub += 1
        if not made:
            made = [(r, x, y, c, "snapped")
                    for r, x, y, c in pins_for(cls, bbox, orient, dark=dark)]
        for role, px, py, pc, src in made:
            g.pins.append(Pin(len(g.pins), i, role, px, py, src, pc))
    g.validate()
    print(g.summary())
    print(f"orientations flipped by wire scoring: {n_flip}")
    print(f"ICs using stub pins: {n_stub}")

    im = gray.convert("RGB")
    d = ImageDraw.Draw(im)
    for p in g.pins:
        r = 7
        col = "lime" if p.source == "stub" else "red"
        d.ellipse([p.x - r, p.y - r, p.x + r, p.y + r], fill=col, outline="yellow")
    im.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
