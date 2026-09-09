"""Rescue demo v2: abnormality proposals + zoom re-infer + low-conf harvest."""
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
from circuitvision.anomaly_rescue import (wire_endpoints, propose_boxes, thick_symbol_boxes,
                            merge_proposals, second_pass, low_conf_harvest)

WEIGHTS = _ROOT / "control" / "runs" / "round3_68_v1" / "best.pt"
SRC = _ROOT / "demo" / "inputs" / "smps.jpg"
OUT = _ROOT / "demo" / "outputs" / "smps_rescue.jpg"
OUT_WIRES = _ROOT / "demo" / "outputs" / "smps_wires_rescued.jpg"


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    res = model.predict(source=str(SRC), conf=0.35, device="cpu", verbose=False)[0]
    dets = dedup_components([(res.names[int(b.cls)], float(b.conf),
                              tuple(float(v) for v in b.xyxy[0].tolist()))
                             for b in res.boxes], 0.5)
    gray = Image.open(SRC).convert("L")
    wire = binarize(gray)
    boxes0 = [b for _, _, b in dets]
    masked = mask_interiors(wire, boxes0)
    dots = find_dots(masked)
    eps = wire_endpoints(masked)
    print(f"endpoints: {len(eps)}, dots: {len(dots)}")

    props_ep, n_unexp = propose_boxes(eps, boxes0, dots, gray.size)
    unexp_pts = [p for p in eps
                 if all(not (b[0] - 20 <= p[0] <= b[2] + 20 and b[1] - 20 <= p[1] <= b[3] + 20) for b in boxes0)
                 and all((p[0] - dx) ** 2 + (p[1] - dy) ** 2 > 14 * 14 for dx, dy, _ in dots)]
    thick = thick_symbol_boxes(masked, boxes0, dots, gray.size)
    props = merge_proposals(props_ep, thick)
    print(f"unexplained endpoints: {n_unexp}, props_endpoint: {len(props_ep)}, "
          f"props_thick: {len(thick)}, props_merged: {len(props)}")

    added_zoom = second_pass(model, gray, props, dets, conf=0.2)
    print(f"recovered_zoom: {len(added_zoom)}")
    for cls, conf, (x0, y0, x1, y1) in sorted(added_zoom, key=lambda a: -a[1])[:15]:
        print(f"  z + {cls} {conf:.2f} ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    mid = dets + added_zoom
    harvested = low_conf_harvest(model, gray, masked, unexp_pts, thick, mid, conf=0.15)
    print(f"recovered_harvest: {len(harvested)}")
    for cls, conf, (x0, y0, x1, y1) in sorted(harvested, key=lambda a: -a[1])[:30]:
        print(f"  h + {cls} {conf:.2f} ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    # dedup harvest vs zoom (IoU>=0.5 keep higher conf)
    from circuitvision.anomaly_rescue import _iou
    final = list(added_zoom)
    for h in harvested:
        dup = [f for f in final if _iou(h[2], f[2]) >= 0.5]
        if not dup:
            final.append(h)
        elif h[1] > max(f[1] for f in dup):
            for f in dup:
                final.remove(f)
            final.append(h)
    print(f"recovered_total: {len(final)} (base {len(dets)} -> {len(dets) + len(final)})")

    im = gray.convert("RGB")
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1, _ in props:
        d.rectangle([x0, y0, x1, y1], outline="orange", width=3)
    for cls, conf, (x0, y0, x1, y1) in final:
        d.rectangle([x0, y0, x1, y1], outline="lime", width=3)
    im.save(OUT)
    print("saved", OUT)

    # cleaned wires: re-mask with rescued boxes, show what Stage 4 now sees
    import cv2
    all_boxes = boxes0 + [b for _, _, b in final]
    masked2 = mask_interiors(wire, all_boxes)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(masked2, 8)
    show = np.zeros_like(masked2)
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            show[lab == i] = 1
    im2 = Image.fromarray((1 - show) * 255).convert("RGB")
    d2 = ImageDraw.Draw(im2)
    for cls, conf, (x0, y0, x1, y1) in final:
        d2.rectangle([x0, y0, x1, y1], outline="lime", width=2)
    im2.save(OUT_WIRES)
    print("saved", OUT_WIRES)


if __name__ == "__main__":
    main()
