"""Shared end-to-end pipeline: detect -> rescue -> loops -> pins -> nets.

Returns everything downstream stages need; keeps stage scripts thin and
comparable (same graph everywhere).
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
from graph_schema import dedup_components, Component, Pin, CircuitGraph
from pin_templates import pins_for, score_orientation, ic_stub_pins, ICS
from wire_nets import extract_nets, binarize, mask_interiors, find_dots
from anomaly_rescue import (wire_endpoints, propose_boxes, thick_symbol_boxes,
                            merge_proposals, second_pass, low_conf_harvest,
                            _iou)
from loop_detector import contour_loops, ic_footprints

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / 'control' / 'runs' / 'round3_68_v1' / 'best.pt'
SRC = HERE / 'demo' / 'inputs' / 'smps.jpg'
_model = None


def model():
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS))
    return _model


def build_graph(conf_base=0.35, conf_zoom=0.2, conf_harvest=0.15,
                verbose=True, src=None):
    m = model()
    src = Path(src) if src is not None else SRC
    res = m.predict(source=str(src), conf=conf_base, device='cpu',
                    verbose=False)[0]
    dets = dedup_components([(res.names[int(b.cls)], float(b.conf),
                              tuple(float(v) for v in b.xyxy[0].tolist()))
                             for b in res.boxes], 0.5)
    gray = Image.open(src).convert('L')
    wire = binarize(gray)
    boxes0 = [b for _, _, b in dets]
    masked = mask_interiors(wire, boxes0)
    dots = find_dots(masked)
    eps = wire_endpoints(masked)
    props_ep, _ = propose_boxes(eps, boxes0, dots, gray.size)
    unexp = [p for p in eps
             if all(not (b[0] - 20 <= p[0] <= b[2] + 20 and b[1] - 20 <= p[1] <= b[3] + 20)
                    for b in boxes0)
             and all((p[0] - dx) ** 2 + (p[1] - dy) ** 2 > 14 * 14
                     for dx, dy, _ in dots)]
    thick = thick_symbol_boxes(masked, boxes0, dots, gray.size)
    props = merge_proposals(props_ep, thick)
    z = second_pass(m, gray, props, dets, conf=conf_zoom)
    mid = dets + z
    h = low_conf_harvest(m, gray, masked, unexp, thick, mid,
                         conf=conf_harvest)
    final = list(z)
    for b in h:
        dup = [f for f in final if _iou(b[2], f[2]) >= 0.5]
        if not dup:
            final.append(b)
        elif b[1] > max(f[1] for f in dup):
            for f in dup:
                final.remove(f)
            final.append(b)
    loops = contour_loops(masked, boxes0, dots, gray.size)
    fps, _ = ic_footprints(masked, boxes0, gray.size)
    if verbose:
        print(f'base={len(dets)} rescued={len(final)} loops={len(loops)} '
              f'footprints={len(fps)}')

    all_dets = dets + final
    dark = np.array(gray) < 128
    g = CircuitGraph()
    for i, (cls, conf, bb) in enumerate(all_dets):
        x0, y0, x1, y1 = bb
        base = 0 if (x1 - x0) >= (y1 - y0) else 90
        orient, score, margin = score_orientation(cls, bb, dark)
        if not (margin >= 1.5 and score > 0.1):
            orient = base
        g.components.append(Component(i, cls, conf, bb, orient))
        made = [(r, x, y, c, 'stub') for r, x, y, c in ic_stub_pins(bb, dark)] \
            if cls in ICS else []
        if not made:
            made = [(r, x, y, c, 'template')
                    for r, x, y, c in pins_for(cls, bb, orient, dark=dark)]
        for role, px, py, pc, src in made:
            g.pins.append(Pin(len(g.pins), i, role, px, py, src, pc))

    boxes = [c.bbox for c in g.components]
    pinxy = [(p.x, p.y) for p in g.pins]
    comp_ids = [p.comp_id for p in g.pins]
    r = extract_nets(gray, boxes, pinxy, comp_ids=comp_ids)
    comps = [(c.cls, c.bbox) for c in g.components]
    pins = [(p.comp_id, p.role, r['frag_of_pin'].get(pi))
            for pi, p in enumerate(g.pins)]
    pin_xy = [(p.x, p.y) for p in g.pins]
    nets = {}
    for pi, ni in r['frag_of_pin'].items():
        nets.setdefault(ni, []).append(pi)
    masked2 = mask_interiors(wire, boxes)
    wire_ctx = {'masked': masked2, 'labels': r['labels'],
                'keep': set(r['net_of']) | {0}, 'net_of': r['net_of'],
                'net_index': r['net_index'], 'dots': r['dots'],
                'boxes': boxes,
                'dark': (np.array(gray) < 128).astype(np.uint8),
                'wire': wire}
    return {'graph': g, 'comps': comps, 'pins': pins, 'pin_xy': pin_xy,
            'nets': nets, 'wire_ctx': wire_ctx, 'gray': gray,
            'footprints': fps, 'extract': r}
