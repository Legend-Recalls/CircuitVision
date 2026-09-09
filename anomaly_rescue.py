"""Anomaly rescue v2: missed components from wire-map abnormalities + re-inference.

Theory: wires are linear (1-3px straight runs). Anything else on the wire
map is a missed component until proven otherwise:
  A. dead-end endpoints unexplained by boxes/dots (a missed axial leaves 2
     ends facing each other across a gap -> min_pts=2, not 3).
  B. thick blobs: morphological opening kills thin wires, survivors (minus
     dots/boxes) are filled symbols — measured 8/9 unexplained on SMPS
     (ground bars). Hollow symbols (diode triangles, coils) are thin-line
     and slip past B; they are caught by the corroborated low-conf harvest.
  C. corroborated low-conf harvest: full-image infer at 0.15, keep only
     novel boxes (IoU<0.5 vs existing) that sit on wire abnormality
     (endpoint / thick-blob / wire-density proximity). This gates
     hallucinations while recovering symbols the crops never cover
     (measured +26 novel on SMPS, incl. the lower-right diode + grounds).

Pipeline: endpoints -> proposals(A) + thick(B) -> merged -> zoomed
second_pass on crops -> full-image low-conf harvest(C) -> merged novel.
"""
import cv2
import numpy as np


def wire_endpoints(wire):
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = cv2.filter2D(wire, -1, k)
    ys, xs = np.nonzero((wire > 0) & (nb == 1))
    return list(zip(xs.tolist(), ys.tolist()))


def _near_box(x, y, boxes, margin):
    for x0, y0, x1, y1 in boxes:
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            return True
    return False


def _near_dot(x, y, dots, r=14):
    r2 = r * r
    for dx, dy, _ in dots:
        if (x - dx) ** 2 + (y - dy) ** 2 <= r2:
            return True
    return False


def propose_boxes(endpoints, boxes, dots, img_wh, margin=20, grid=80,
                   min_pts=2, box_margin=50, max_frac=0.25):
    W, H = img_wh
    pts = [p for p in endpoints
           if not _near_box(*p, boxes, margin) and not _near_dot(*p, dots)]
    n_unexplained = len(pts)
    cells = {}
    for x, y in pts:
        cells.setdefault((x // grid, y // grid), []).append((x, y))
    parent = {c: c for c in cells}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    keys = list(cells)
    nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for cx, cy in keys:
        for dx, dy in nb8:
            nb = (cx + dx, cy + dy)
            if nb in parent:
                union((cx, cy), nb)
    groups = {}
    for c in keys:
        groups.setdefault(find(c), []).extend(cells[c])
    props = []
    for gpts in groups.values():
        if len(gpts) < min_pts:
            continue
        xs = [p[0] for p in gpts]
        ys = [p[1] for p in gpts]
        x0, y0 = max(0, min(xs) - box_margin), max(0, min(ys) - box_margin)
        x1, y1 = min(W, max(xs) + box_margin), min(H, max(ys) + box_margin)
        if (x1 - x0) * (y1 - y0) > max_frac * W * H:
            continue
        props.append((x0, y0, x1, y1, len(gpts)))
    return props, n_unexplained


def thick_symbol_boxes(masked, boxes, dots, img_wh, area_min=60,
                       dot_r=20, box_margin=8, pad=30):
    """Opening kills thin (1-3px) wires; survivors are dots + filled symbols.

    Returns proposal boxes for survivors unexplained by dots/boxes.
    masked: uint8 0/1 HxW. img_wh: (W, H).
    """
    import cv2
    W, H = img_wh
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(masked.astype(np.uint8), cv2.MORPH_OPEN, k5)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(opened, 8)
    props = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_min:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        if _near_dot(cx, cy, dots, r=dot_r):
            continue
        if _near_box(cx, cy, boxes, box_margin):
            continue
        x0, y0, w, hh = (int(stats[i, cv2.CC_STAT_LEFT]),
                         int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]),
                         int(stats[i, cv2.CC_STAT_HEIGHT]))
        props.append((max(0, x0 - pad), max(0, y0 - pad),
                      min(W, x0 + w + pad), min(H, y0 + hh + pad),
                      int(stats[i, cv2.CC_STAT_AREA])))
    return props


def merge_proposals(a, b, iou_thr=0.3):
    """Union of two proposal lists, deduping overlaps (keep higher count)."""
    out = list(a)
    for p in b:
        if all(_iou(p[:4], q[:4]) < iou_thr for q in out):
            out.append(p)
        else:
            j = max(range(len(out)), key=lambda k: _iou(p[:4], out[k][:4]))
            if p[4] > out[j][4]:
                out[j] = p
    return out


def _corroborated(bbox, unexp_pts, thick, masked, img_wh, pt_margin=30):
    """A novel box is kept only if it sits on wire abnormality."""
    x0, y0, x1, y1 = bbox
    for px, py in unexp_pts:
        if x0 - pt_margin <= px <= x1 + pt_margin and \
                y0 - pt_margin <= py <= y1 + pt_margin:
            return True
    for tx0, ty0, tx1, ty1, _ in thick:
        if not (x1 < tx0 or x0 > tx1 or y1 < ty0 or y0 > ty1):
            return True
    H, W = masked.shape
    xa, xb = max(0, int(x0)), min(W, int(x1))
    ya, yb = max(0, int(y0)), min(H, int(y1))
    if xb > xa and yb > ya and int(masked[ya:yb, xa:xb].sum()) >= 20:
        return True
    return False


def low_conf_harvest(model, gray_pil, masked, unexp_pts, thick, existing,
                     conf=0.15, device="cpu"):
    """Full-image infer at low conf; keep novel boxes on abnormalities."""
    import numpy as np
    W, H = gray_pil.size
    r = model.predict(source=np.array(gray_pil.convert("RGB")), conf=conf,
                      device=device, verbose=False)[0]
    if r.boxes is None:
        return []
    img_wh = (W, H)
    _ = img_wh
    added = []
    for b in r.boxes:
        bx0, by0, bx1, by1 = [float(v) for v in b.xyxy[0].tolist()]
        g = (bx0 * W / r.orig_shape[1], by0 * H / r.orig_shape[0],
             bx1 * W / r.orig_shape[1], by1 * H / r.orig_shape[0]) \
            if r.orig_shape[1] != W or r.orig_shape[0] != H else \
            (bx0, by0, bx1, by1)
        if any(_iou(g, e[2]) >= 0.5 for e in existing + added):
            continue
        if _corroborated(g, unexp_pts, thick, masked, (W, H)):
            added.append((r.names[int(b.cls)], float(b.conf), g))
    return added


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def second_pass(model, gray_pil, proposals, existing, conf=0.2, imgsz=640,
                merge_iou=0.5, pad=30):
    """Run inference zoomed on proposal crops; keep boxes novel vs existing.

    existing: [(cls, conf, bbox)]. Returns added [(cls, conf, bbox_global)].
    pad: extra context around each proposal (tight endpoint clusters cut
    symbol edges; context restores them).
    """
    W, H = gray_pil.size
    added = []
    for x0, y0, x1, y1 in [p[:4] for p in proposals]:
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        crop = gray_pil.crop((int(x0), int(y0), int(x1), int(y1)))
        s = imgsz / max(crop.size)
        crop_up = crop.resize((int(crop.size[0] * s), int(crop.size[1] * s)))
        r = model.predict(source=np.array(crop_up.convert("RGB")), conf=conf,
                          device="cpu", verbose=False)[0]
        if r.boxes is None:
            continue
        for b in r.boxes:
            bx0, by0, bx1, by1 = [float(v) for v in b.xyxy[0].tolist()]
            g = (x0 + bx0 / s, y0 + by0 / s, x0 + bx1 / s, y0 + by1 / s)
            if all(_iou(g, e[2]) < merge_iou for e in existing + added):
                added.append((r.names[int(b.cls)], float(b.conf), g))
    return added
