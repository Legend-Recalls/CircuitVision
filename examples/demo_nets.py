"""Demo Stage 4: detect -> pins -> nets -> color overlay + metrics.

Run from D:/migration/new/circuitvision. CPU only.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
from circuitvision.graph_schema import Component, Pin, CircuitGraph, dedup_components
from circuitvision.pin_templates import pins_for, score_orientation, ic_stub_pins, ICS
from circuitvision.wire_nets import extract_nets

WEIGHTS = _ROOT / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = _ROOT / "demo" / "inputs" / "smps.jpg"
OUT = _ROOT / "demo" / "outputs" / "smps_nets.jpg"
CONF = 0.35


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=CONF, device="cpu", verbose=False)[0]
    raw = [(res.names[int(b.cls)], float(b.conf),
            tuple(float(v) for v in b.xyxy[0].tolist())) for b in res.boxes]
    dets = dedup_components(raw, 0.5)

    gray = Image.open(SRC).convert("L")
    dark = np.array(gray) < 128

    g = CircuitGraph()
    for i, (cls, conf, bbox) in enumerate(dets):
        x0, y0, x1, y1 = bbox
        base = 0 if (x1 - x0) >= (y1 - y0) else 90
        orient, score, margin = score_orientation(cls, bbox, dark)
        if not (margin >= 1.5 and score > 0.1):
            orient = base
        g.components.append(Component(i, cls, conf, bbox, orient))
        made = []
        if cls in ICS:
            stubs = ic_stub_pins(bbox, dark)
            if stubs:
                made = [(r, x, y, c, "stub") for r, x, y, c in stubs]
        if not made:
            made = [(r, x, y, c, "template")
                    for r, x, y, c in pins_for(cls, bbox, orient, dark=dark)]
        for role, px, py, pc, src in made:
            g.pins.append(Pin(len(g.pins), i, role, px, py, src, pc))

    boxes = [c.bbox for c in g.components]
    pinxy = [(p.x, p.y) for p in g.pins]
    r = extract_nets(gray, boxes, pinxy)
    g.nets = sorted(set(r["frag_of_pin"].values()))
    for pi, ni in r["frag_of_pin"].items():
        g.edges_pin_net.append((pi, ni))
    g.validate()
    print(g.summary())
    print("stage4:", r["stats"])

    # overlay: color ONLY pin-bearing nets (the 65 that matter); everything
    # else stays grayscale so text/noise fragments don't rainbow the sheet.
    import cv2
    labels = r["labels"]
    pinned_frags = set()
    for frag, root in r["net_of"].items():
        # net_of holds union-find roots; g.nets holds compact indices.
        if r["net_index"][root] in g.nets:
            pinned_frags.add(frag)
    lab_color = np.zeros((*labels.shape, 3), np.uint8)
    palette = np.random.default_rng(7).integers(40, 255, size=(len(g.nets), 3))
    net_of_pinnet = {n: i for i, n in enumerate(g.nets)}
    keep_mask = np.isin(labels, list(pinned_frags))
    for frag in pinned_frags:
        lab_color[labels == frag] = palette[net_of_pinnet[r["net_index"][r["net_of"][frag]]] % len(palette)]
    base = np.array(gray.convert("RGB"))
    overlay = np.where(keep_mask[..., None], lab_color, base).astype(np.uint8)
    im = Image.fromarray(overlay)
    d = ImageDraw.Draw(im)
    for dx, dy, _ in r["dots"]:
        d.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], outline="white", width=2)
    for pi in r["isolated"]:
        p = g.pins[pi]
        d.ellipse([p.x - 10, p.y - 10, p.x + 10, p.y + 10], outline="magenta", width=3)
    # net index labels on the largest nets so nets read as entities.
    # Everything here is in compact index space (g.nets), never raw roots.
    def frag_index(frag):
        return r["net_index"][r["net_of"][frag]]

    sizes = {}
    for frag in pinned_frags:
        ni = frag_index(frag)
        sizes[ni] = sizes.get(ni, 0) + int((labels == frag).sum())
    for ni in sorted(sizes, key=lambda k: -sizes[k])[:12]:
        f0 = next(f for f in pinned_frags if frag_index(f) == ni)
        ys, xs = np.nonzero(labels == f0)
        d.text((int(xs.mean()), int(ys.mean())), f"N{ni}",
               fill="white", stroke_width=2, stroke_fill="black")
    im.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
