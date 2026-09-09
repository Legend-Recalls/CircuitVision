"""Loop-detector: large hollow packages (transistor cans, transformer frames,
IC bodies) survive on the wire map as big closed loops.

Wires rarely form large closed loops (rails form rectangles WITH dots at
corners; loops with no dot and no box = package outline). Two detectors:
  1. contour loops: external contours, area 1500-200000px, circular
     (circularity>0.55, e.g. Q cans) or rectangular (4-vertex approx +
     extent>0.7, e.g. transformer frames). Excludes dot-centered and
     already-boxed loops.
  2. IC footprint rows: HoughCircles finds small terminal circles
     (r 5-16px); >=6 collinear circles with regular spacing and no box
     cover = IC/connector footprint (IC1 KA7500E has no drawn body, only
     a pin row). The classifier is blind there (no fire even at 0.10),
     so footprints are returned with class hint for Stage 7 structural
     recovery; crops are still attempted at 0.05 + TTA.
"""
import cv2
import numpy as np


def contour_loops(masked, boxes, dots, img_wh, area_lo=1500,
                  area_hi=200000, dot_r=20, box_iou=0.3, pad=40):
    from anomaly_rescue import _iou, _near_dot, _near_box
    W, H = img_wh
    cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    props = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (area_lo <= area <= area_hi):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        if _near_dot(cx, cy, dots, r=dot_r):
            continue
        if _near_box(cx, cy, boxes, 8):
            continue
        circ = 4 * np.pi * area / (peri * peri)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        extent = area / (w * h) if w * h > 0 else 0
        is_circle = circ > 0.55 and 0.7 <= w / max(h, 1) <= 1.43
        is_rect = len(approx) == 4 and extent > 0.7 and w > 40 and h > 40
        if not (is_circle or is_rect):
            continue
        kind = 'circle' if is_circle else 'rect'
        if any(_iou((x - pad, y - pad, x + w + pad, y + h + pad),
                    b) >= box_iou for b in boxes):
            continue
        props.append((max(0, x - pad), max(0, y - pad),
                      min(W, x + w + pad), min(H, y + h + pad),
                      kind, round(circ, 2)))
    return props


def ic_footprints(masked, boxes, img_wh, min_n=5, span_min=150, pad=50):
    """Lines of small terminal circles = IC/connector footprint.

    IC1 KA7500E has no drawn body: top pin row (horizontal) + right pin
    column (vertical). So detect both orientations, then union overlapping
    groups (L-shapes merge into one footprint).
    The classifier is blind here (no fire even at 0.10), so footprints
    carry class hint 'integrated_circuit' for Stage 7 structural recovery;
    crops are still attempted at 0.05 + TTA.
    """
    from anomaly_rescue import _iou
    W, H = img_wh
    blur = cv2.medianBlur((masked * 255).astype(np.uint8), 3)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=15,
                               minRadius=5, maxRadius=16)
    if circles is None:
        return [], []
    cs = [(float(x), float(y), float(r)) for x, y, r in circles[0]]
    # keep circles unexplained by boxes
    free = [c for c in cs
            if not any(b[0] - 6 <= c[0] <= b[2] + 6 and
                       b[1] - 6 <= c[1] <= b[3] + 6 for b in boxes)]
    # horizontal rows (cluster y) + vertical columns (cluster x)
    groups = []
    for axis in (1, 0):
        order = sorted(range(len(free)), key=lambda i: free[i][axis])
        used = [False] * len(free)
        for i in order:
            if used[i]:
                continue
            grp = [j for j in range(len(free))
                   if abs(free[j][axis] - free[i][axis]) <= 14]
            if len(grp) >= min_n:
                other = 1 - axis
                xs = sorted(free[j][other] for j in grp)
                if xs[-1] - xs[0] >= span_min:
                    groups.append(set(grp))
                    for j in grp:
                        used[j] = True
    # union overlapping groups (L-shaped pin arrangements)
    merged = []
    for g in groups:
        for m in merged:
            if g & m:
                m |= g
                break
        else:
            merged.append(set(g))
    props = []
    for grp in merged:
        xs = [free[j][0] for j in grp]
        ys = [free[j][1] for j in grp]
        x0, x1 = max(0, int(min(xs)) - pad), min(W, int(max(xs)) + pad)
        # body extends below the top pin row (DIP-style): cover lower pins
        y0, y1 = max(0, int(min(ys)) - 60), min(H, int(max(ys)) + 260)
        if any(_iou((x0, y0, x1, y1), b) >= 0.5 for b in boxes):
            continue
        props.append((x0, y0, x1, y1, len(grp)))
    return props, free
