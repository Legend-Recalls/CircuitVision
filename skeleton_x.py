"""Skeleton (Zhang-Suen) + crossing inventory for Stage 5 check 5.

Convention: T-junction = connected; +-crossing: dot = connected,
no dot = hop-over (must stay split). This module finds skeleton
degree-4 points (X-crossings) outside dots/boxes on substantial copper.
"""
import numpy as np


def zhang_suen(img):
    """img: uint8 0/1. Returns skeleton uint8 0/1. Vectorized numpy."""
    sk = img.copy().astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(sk, 1)
            p2 = P[:-2, 1:-1]
            p3 = P[:-2, 2:]
            p4 = P[1:-1, 2:]
            p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]
            p7 = P[2:, :-2]
            p8 = P[1:-1, :-2]
            p9 = P[:-2, :-2]
            nbrs = [p2, p3, p4, p5, p6, p7, p8, p9]
            n1 = sum(nbrs)
            seq = sum((a == 0) & (b == 1)
                      for a, b in zip(nbrs, nbrs[1:] + nbrs[:1]))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            kill = (sk == 1) & (n1 >= 2) & (n1 <= 6) & (seq == 1) & cond
            if kill.any():
                sk[kill] = 0
                changed = True
        if not changed:
            return sk


def skeleton_degree(skel):
    """8-neighbor count on 1px skeleton."""
    m = skel.astype(np.int32)
    h, w = m.shape
    neigh = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neigh[1:-1, 1:-1] += m[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
    return neigh


def all_circles(wire_img, dots=(), dot_r=14):
    """Hough circles r 5-60px on the UNMASKED wire map (masking cuts
    terminal circles near rescued boxes, e.g. T1) minus validated dots."""
    import cv2
    blur = cv2.medianBlur((wire_img * 255).astype(np.uint8), 3)
    out = []
    for maxr in (20, 60):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=15,
                             minRadius=5, maxRadius=maxr)
        if c is not None:
            out += [(float(x), float(y), float(r)) for x, y, r in c[0]]
    nodot, seen = [], set()
    for cx, cy, r in out:
        if any((cx - dx) ** 2 + (cy - dy) ** 2 <= dot_r * dot_r
               for dx, dy, _ in dots):
            continue
        key = (round(cx / 8), round(cy / 8))
        if key in seen:
            continue
        seen.add(key)
        nodot.append((cx, cy, r))
    return nodot


def ring_runs(masked, x, y, r=11, samples=64):
    """Distinct wire arms crossing a ring: T-tap=3, X-crossing=4."""
    import math
    h, w = masked.shape
    vals = []
    for t in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        xx, yy = int(round(x + r * math.cos(t))), int(round(y + r * math.sin(t)))
        vals.append(masked[yy, xx] if 0 <= xx < w and 0 <= yy < h else 0)
    runs, in_run = 0, False
    for v in vals + vals[:4]:
        if v and not in_run:
            runs += 1
            in_run = True
        elif not v:
            in_run = False
    return runs


def explain_x(xs, dark, boxes, circles, labels=None, areas=None,
              masked=None, fill_thr=0.6, box_dist=40):
    """Partition skeleton X's.

    Drafting convention (the decisive rule): a T (3 arms) is a connection
    with or without a dot glyph — only 4-arm crossings are ambiguous.
    So: fill>=thr or arms<=3 -> junction (fine); bare 4-arm X's explained
    by symbol geometry (near box / on circle) or text strokes; the rest
    are hop-over candidates for Stage 7.

    Returns dict {dotted, explained, text, unexplained} of (x, y, fill).
    """
    h, w = dark.shape
    yy, xx = np.mgrid[:h, :w]
    dotted, explained, text, unexplained = [], [], [], []
    for x, y, _ in xs:
        xi, yi = int(round(x)), int(round(y))
        disc = (xx - xi) ** 2 + (yy - yi) ** 2 <= 25
        fill = round(float(dark[disc].mean()), 2)
        arms = ring_runs(masked, xi, yi) if masked is not None else 4
        if fill >= fill_thr or arms <= 3:
            dotted.append((x, y, fill))
            continue
        near_box = any(b[0] - box_dist <= x <= b[2] + box_dist and
                       b[1] - box_dist <= y <= b[3] + box_dist
                       for b in boxes)
        on_circle = any(abs(np.hypot(x - cx, y - cy) - r) <= 15
                        for cx, cy, r in circles)
        # terminal clusters (transformer secondaries: circles + short
        # leads + rails interleave; verified at T1 (1192,875)): within one
        # terminal pitch (~55px) of >=2 circles = cluster geometry.
        near_cluster = sum(np.hypot(x - cx, y - cy) <= 55
                           for cx, cy, _ in circles) >= 2
        if near_box or on_circle or near_cluster:
            explained.append((x, y, fill))
            continue
        if labels is not None and areas is not None:
            x0, x1 = max(0, xi - 10), min(w, xi + 11)
            y0, y1 = max(0, yi - 10), min(h, yi + 11)
            small = any(areas[int(v)] < 200
                        for v in np.unique(labels[y0:y1, x0:x1]) if int(v) != 0)
            if small:
                text.append((x, y, fill))
                continue
        unexplained.append((x, y, fill))
    return {'dotted': dotted, 'explained': explained, 'text': text,
            'unexplained': unexplained}


def x_crossings(masked, labels, areas, dots, boxes, min_area=500,
                dot_r=14, box_margin=10):
    """Degree>=4 skeleton points outside dots/boxes on big fragments.

    Returns (list of (x, y, cluster_px), skeleton, degree_map).
    """
    import cv2
    sk = zhang_suen((masked > 0).astype(np.uint8))
    deg = skeleton_degree(sk)
    h, w = masked.shape
    cand = np.zeros_like(sk)
    ys, xs = np.nonzero((sk > 0) & (deg >= 4))
    for x, y in zip(xs.tolist(), ys.tolist()):
        if areas[labels[y, x]] < min_area:
            continue
        if any((x - dx) ** 2 + (y - dy) ** 2 <= dot_r * dot_r
               for dx, dy, _ in dots):
            continue
        if any(b[0] - box_margin <= x <= b[2] + box_margin and
               b[1] - box_margin <= y <= b[3] + box_margin for b in boxes):
            continue
        cand[y, x] = 1
    # dilate slightly so multi-px crossing cores cluster as one
    cand = cv2.dilate(cand, np.ones((5, 5), np.uint8))
    nc, _, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    out = [(round(float(cent[i][0]), 1), round(float(cent[i][1]), 1),
            int(stats[i, cv2.CC_STAT_AREA]))
           for i in range(1, nc)]
    return out, sk, deg
