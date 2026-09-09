"""Stage 4 v1: wire/net extraction with classical CV (no learning).

Pipeline: binarize -> mask component interiors -> connected components
(fragments) -> junction-dot merge (union-find) -> pin-to-net assignment.

Topology rules (from roadmap):
- Fragments touching the same junction DOT merge (dots = connections).
- Crossings WITHOUT a dot stay separate (no-connect).
- Text strokes die by area filter.
- Pins attach to the nearest wire within a radius; the rest are flagged
  isolated for Stage 7 (graph correction), not silently dropped.
"""
import cv2
import numpy as np


def binarize(gray, thresh=None):
    """Dark-pixel mask. Default is Otsu (per-image adaptive): black CAD
    and light-blue CAD both split correctly (measured Otsu: smps 138,
    wien 156, cir1 173). A fixed 128 dropped whole thin-wire runs on
    cir1 (50-100px gaps, 2 opens). Pass thresh to pin the old behavior.
    """
    import cv2
    arr = np.array(gray)
    if thresh is None:
        thresh, _ = cv2.threshold(arr, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (arr < float(thresh)).astype(np.uint8)


def mask_interiors(wire, boxes, dilate=3):
    """Zero out component interiors so symbols don't bridge nets."""
    m = wire.copy()
    h, w = m.shape
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0) - dilate), max(0, int(y0) - dilate)
        x1, y1 = min(w, int(x1) + dilate), min(h, int(y1) + dilate)
        m[y0:y1, x0:x1] = 0
    return m


def find_dots(wire, min_area=12, max_area=400):
    """Filled junction dots: small solid blobs (high circularity) WITH wires.

    Letters pass the shape test, so every candidate must also show >=2 wire
    crossings on an annulus around it (real junctions join wires; letters
    touch nothing). Mid-wire dots give exactly 2, crossings 3-4.
    """
    cnts, _ = cv2.findContours(wire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    h, w = wire.shape
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        if circ <= 0.55:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        # annulus crossings at r=9: >=2 dark runs => wires leave the dot
        r = 9
        ring = [wire[int(round(cy + r * np.sin(t))), int(round(cx + r * np.cos(t)))]
                for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)
                if 0 <= int(round(cy + r * np.sin(t))) < h
                and 0 <= int(round(cx + r * np.cos(t))) < w]
        runs, in_run = 0, False
        for v in ring + ring[:4]:
            if v and not in_run:
                runs += 1
                in_run = True
            elif not v:
                in_run = False
        if runs >= 2:
            dots.append((cx, cy, area))
    return dots


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        p = self.p.setdefault(a, a)
        while self.p[p] != p:
            self.p[p] = self.p[self.p[p]]
            p = self.p[p]
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def astar_to_wire(wire, start, target_ids, labels, window=240, gap_cost=12.0,
                  max_expand=60000):
    """A* from pin to the nearest wire pixel of a *different* fragment.

    wire: bool HxW (True = copper). Background costs gap_cost (bridges small
    breaks from masking/thresholding); wire costs 1. 8-connected, windowed
    around start for speed. Returns (frag_id, path_cost) or (None, inf).

    Honest note: with the goal being "any wire pixel" (location unknown),
    the A* heuristic is zero and this reduces to Dijkstra/Lee expansion —
    the same maze-routing family maps apps use, minus the goal heuristic
    (there is no single goal to aim at). PCB autorouters do exactly this.
    """
    import heapq

    h, w = wire.shape
    sx, sy = int(start[0]), int(start[1])
    x0, x1 = max(0, sx - window), min(w, sx + window + 1)
    y0, y1 = max(0, sy - window), min(h, sy + window + 1)
    if x1 <= x0 or y1 <= y0:
        return None, float("inf")
    sub_w = wire[y0:y1, x0:x1]
    sub_l = labels[y0:y1, x0:x1]
    lx, ly = sx - x0, sy - y0

    INF = float("inf")
    g = {(lx, ly): 0.0}
    pq = [(0.0, 0.0, (lx, ly))]  # h=0: goal location unknown -> Dijkstra
    expanded = 0
    NB = [(-1, -1, 1.4142), (0, -1, 1.0), (1, -1, 1.4142),
          (-1, 0, 1.0), (1, 0, 1.0),
          (-1, 1, 1.4142), (0, 1, 1.0), (1, 1, 1.4142)]
    H, W = sub_w.shape
    while pq and expanded < max_expand:
        _, gc, (x, y) = heapq.heappop(pq)
        if gc > g.get((x, y), INF):
            continue
        expanded += 1
        if (x, y) != (lx, ly) and sub_w[y, x] and int(sub_l[y, x]) in target_ids:
            return int(sub_l[y, x]), gc
        for dx, dy, step in NB:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            step_cost = step * (1.0 if sub_w[ny, nx] else gap_cost)
            ng = gc + step_cost
            if ng < g.get((nx, ny), INF):
                g[(nx, ny)] = ng
                heapq.heappush(pq, (ng, ng, (nx, ny)))
    return None, INF


def extract_nets(gray, boxes, pins, dot_radius=10, pin_radius=12, min_frag=15,
                 comp_ids=None):
    """Full Stage-4 pass.

    boxes: [(x0,y0,x1,y1)]. pins: [(x, y, ...)] (coords used).
    comp_ids: optional parallel list pin -> component idx. When given, the
    maze rescue refuses to attach a pin to a net already held by a sibling
    pin of the same component (that path would fake a SHORTED part: the
    search conducts through the component's own unmasked body).
    Returns dict: labels map, nets {net_id: fragment ids}, frag_of_pin,
    isolated pin idxs, dots, stats.
    """
    wire = binarize(gray)
    masked = mask_interiors(wire, boxes)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(masked, 8)
    # drop text-stroke fragments by area (keep id remap)
    keep = {0}
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] >= min_frag:
            keep.add(i)
    uf = UnionFind()
    dots = find_dots(masked)
    h, w = masked.shape
    # fragment id of each dot-touching pixel -> union
    dot_frags = []
    for dx, dy, _ in dots:
        r = dot_radius
        x0, x1 = max(0, int(dx) - r), min(w, int(dx) + r + 1)
        y0, y1 = max(0, int(dy) - r), min(h, int(dy) + r + 1)
        ids = set(int(v) for v in np.unique(labels[y0:y1, x0:x1]) if v in keep)
        dot_frags.append(sorted(ids))
        for a in ids:
            for b in ids:
                uf.union(a, b)
    # canonical net per kept fragment
    net_of = {}
    for i in keep:
        if i == 0:
            continue
        net_of[i] = uf.find(i)
    net_ids = sorted(set(net_of.values()))
    net_index = {n: i for i, n in enumerate(net_ids)}

    frag_of_pin, isolated = {}, []
    for pi, (px, py, *_) in enumerate(pins):
        x0, x1 = max(0, int(px) - pin_radius), min(w, int(px) + pin_radius + 1)
        y0, y1 = max(0, int(py) - pin_radius), min(h, int(py) + pin_radius + 1)
        ids = [int(v) for v in np.unique(labels[y0:y1, x0:x1])
               if v in keep and v != 0]
        if not ids:
            isolated.append(pi)
            continue
        # nearest fragment by distance to pin
        best, best_d = None, 1e9
        yy, xx = np.nonzero(np.isin(labels[y0:y1, x0:x1], ids))
        for dx, dy in zip(xx, yy):
            d = (dx + x0 - px) ** 2 + (dy + y0 - py) ** 2
            if d < best_d:
                best_d = d
                best = int(labels[y0 + dy, x0 + dx])
        frag_of_pin[pi] = net_index[net_of[best]]

    # Maze-routing rescue: isolated pins get a Dijkstra search over the
    # UNMASKED wire map (masked interiors still conduct — stubs live there).
    # Accepts bridges up to ~15px of gap; failures stay isolated for Stage 7.
    # Sibling-net exclusion (needs comp_ids): never attach to a net held by
    # a sibling pin — that route conducts through the part's own body and
    # manufactures SHORTED two-terminals (Stage 5 caught 25 of these).
    bridged = {}
    targets = set(net_of)
    sib_nets = {}
    if comp_ids is not None:
        comp_nets = {}
        for pi, ci in enumerate(comp_ids):
            if pi in frag_of_pin:
                comp_nets.setdefault(ci, set()).add(frag_of_pin[pi])
        for pi, ci in enumerate(comp_ids):
            sib_nets[pi] = comp_nets.get(ci, set())
    for pi in list(isolated):
        px, py, *_ = pins[pi]
        allow = targets - {f for f in targets
                           if net_index[net_of[f]] in sib_nets.get(pi, set())}
        if not allow:
            continue  # only sibling nets nearby: attaching shorts the part
        frag, cost = astar_to_wire(wire, (px, py), allow, labels)
        if frag is not None and cost <= 200.0:
            frag_of_pin[pi] = net_index[net_of[frag]]
            isolated.remove(pi)
            bridged[pi] = round(cost, 1)
    return {
        "labels": labels, "net_of": net_of, "net_index": net_index,
        "frag_of_pin": frag_of_pin, "isolated": isolated,
        "bridged": bridged,
        "dots": dots, "dot_frags": dot_frags,
        "stats": {
            "fragments": len(net_of), "nets": len(net_ids),
            "pins": len(pins), "assigned": len(frag_of_pin),
            "isolated": len(isolated), "bridged": len(bridged),
            "dots": len(dots),
        },
    }
