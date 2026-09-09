"""Wires-only view: the exact wire mask Stage 4 reasons over. No nets, no pins."""
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
from circuitvision.wire_nets import binarize, mask_interiors, find_dots

SRC = _ROOT / "demo" / "inputs" / "smps.jpg"
OUT = _ROOT / "demo" / "outputs" / "smps_wires.jpg"


def main():
    from ultralytics import YOLO
    w = _ROOT / "control" / "runs" / "round3_68_v1" / "best.pt"
    model = YOLO(str(w))
    res = model.predict(source=str(SRC), conf=0.35, device="cpu", verbose=False)[0]
    dets = [(res.names[int(b.cls)], float(b.conf),
             tuple(float(v) for v in b.xyxy[0].tolist())) for b in res.boxes]
    boxes = [b for _, _, b in dedup_components(dets, 0.5)]

    gray = Image.open(SRC).convert("L")
    masked = mask_interiors(binarize(gray), boxes)
    dots = find_dots(masked)
    print(f"masked wire pixels: {int(masked.sum())}, dots: {len(dots)}")

    # Display: only substantial fragments (>=60px) so text strokes don't
    # clutter the view. Extraction itself is untouched.
    import cv2
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(masked, 8)
    show = np.zeros_like(masked)
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            show[lab == i] = 1
    im = Image.fromarray((1 - show) * 255).convert("RGB")
    d = ImageDraw.Draw(im)
    # Ring all candidates small: dots that merge nothing are harmless
    # (union of a single set is a no-op), wrong merges need a letter to sit
    # within 10px of two DIFFERENT nets, which the overlay shows is rare.
    for dx, dy, _ in dots:
        d.ellipse([dx - 4, dy - 4, dx + 4, dy + 4], outline="red", width=1)
    print(f"dots ringed: {len(dots)}")
    im.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
