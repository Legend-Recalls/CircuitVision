"""Demo: detect (Stage 1) -> pin grounding (Stage 3) on a real schematic.

Run from D:/migration/new/circuitvision. CPU only.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_schema import Component, Pin, CircuitGraph
from pin_templates import pins_for

WEIGHTS = Path(__file__).resolve().parent / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = Path(__file__).resolve().parent / "demo" / "inputs" / "smps.jpg"
OUT = Path(__file__).resolve().parent / "demo" / "outputs" / "smps_pins.jpg"
CONF = 0.35


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=CONF, device="cpu", verbose=False)[0]

    g = CircuitGraph()
    for i, b in enumerate(res.boxes):
        cls = res.names[int(b.cls)]
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        w, h = x1 - x0, y1 - y0
        orient = 0 if w >= h else 90
        g.components.append(Component(i, cls, float(b.conf), (x0, y0, x1, y1), orient))
        for role, px, py, pc in pins_for(cls, (x0, y0, x1, y1), orient):
            g.pins.append(Pin(len(g.pins), i, role, px, py, "template", pc))
    g.validate()
    print(g.summary())
    by_cls = {}
    for c in g.components:
        by_cls[c.cls] = by_cls.get(c.cls, 0) + 1
    print("top classes:", sorted(by_cls.items(), key=lambda kv: -kv[1])[:8])

    im = Image.open(SRC).convert("RGB")
    d = ImageDraw.Draw(im)
    for p in g.pins:
        r = 7
        d.ellipse([p.x - r, p.y - r, p.x + r, p.y + r], fill="red", outline="yellow")
    im.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
