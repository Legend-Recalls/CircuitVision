"""Stage 3: orientation-aware pin templates (rule templates first).

Each kept class maps to a template kind. pins_for() returns absolute pixel
pins for a detection bbox. Coordinates are normalized (0..1) inside the box.
Axial kinds auto-orient from aspect ratio; the rest take an orientation
(0/90/180/270, default 0 = as-drawn) which a future classifier will supply.
Every pin carries a conf so Stage 7 knows what to distrust.
"""

# -- template kinds -----------------------------------------------------------
# axial2  : 2 pins at the ends of the long axis (resistors, caps, diodes...)
# ground1 : 1 pin, connection edge on top (gnd, vss)
# rail1   : 1 pin, connection edge at bottom (vdd)
# port1   : 1 pin, center (port, terminal, probes) — edge analysis later
# bjt3    : collector top, base mid-left, emitter bottom (npn/pnp/bjt)
# fet3    : drain top, gate mid-left, source bottom (mosfet/nmos/pmos...)
# gate    : inputs left edge, output right middle (and/or/not/...)
# opamp   : in-/in+ left, out right middle
# ic      : pins along both vertical edges, pitch-estimated, conf 0.5
# xformer : 2 pins per side (transformer, relay)
# none    : no electrical pins (text, crossover, junction->net node in stage 5)

AXIAL2 = {
    "resistor", "resistor.adjustable", "resistor.photo", "capacitor",
    "capacitor.adjustable", "capacitor.polarized", "inductor",
    "inductor.ferrite", "fuse", "diode", "diode.zener", "diode.light_emitting",
    "lamp", "light", "crystal", "thermistor", "varistor", "heating_element",
    "antenna", "speaker", "microphone", "switch", "diac", "clock",
    "voltimeter", "socket",
}
GROUND1 = {"gnd", "vss"}
RAIL1 = {"vdd"}
PORT1 = {"port", "terminal", "probe", "probe.current", "v", "motor", "generator"}
BJT3 = {"npn", "pnp", "transistor.bjt", "thyristor", "transistor.photo"}
FET3 = {"mosfet", "nmos", "nmos.bulk", "pmos", "pmos.bulk"}
GATES = {"and", "or", "nand", "nor", "xor", "not"}
OPAMP = {"operational_amplifier"}
ICS = {"integrated_circuit", "integrated_circuit.ne555",
       "integrated_circuit.voltage_regulator", "display.7_segment"}
XFORMER = {"transformer", "relay", "optocoupler"}
NONE = {"text", "explanatory", "crossover", "crossover.curved", "junction",
        "block", "optical", "magnetic", "mechanical", "connector", "unknown"}
SOURCES2 = {"voltage", "voltage.ac", "voltage.dc", "voltage.battery",
            "voltage.dc.one_port", "voltage.lines", "current_source",
            "current_source.ac", "current_source.dc", "dependent_current_source"}
TGATE4 = {"tgate"}
AMP1 = {"amplifier.single_end"}


def _rot(pt, orientation):
    x, y = pt
    o = orientation % 360
    if o == 0:
        return (x, y)
    if o == 90:
        return (1 - y, x)
    if o == 180:
        return (1 - x, 1 - y)
    if o == 270:
        return (y, 1 - x)
    raise ValueError(orientation)


def _at(bbox, pt):
    x0, y0, x1, y1 = bbox
    nx, ny = pt
    return (x0 + nx * (x1 - x0), y0 + ny * (y1 - y0))


def _disc_score(dark, px, py, r=5):
    h, w = dark.shape
    x0, x1 = max(0, int(px) - r), min(w, int(px) + r + 1)
    y0, y1 = max(0, int(py) - r), min(h, int(py) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(dark[y0:y1, x0:x1].mean())


def _raw_pins(cls, bbox, orientation=0):
    """Normalized [(role, (nx, ny), conf)] before pixel mapping."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    out = []

    def emit(role, pt, conf=0.9):
        out.append((role, _rot(pt, orientation), conf))

    if cls in AXIAL2 or cls in SOURCES2:
        # Canonical = horizontal; orientation rotates it (90/270 = vertical).
        # The axis is NEVER guessed from aspect here: double-rotation
        # (aspect branch + _rot) once put horizontal pins on vertical parts.
        # Callers pick orientation (aspect prior, refined by score_orientation).
        emit("p0", (0.0, 0.5)); emit("p1", (1.0, 0.5))
    elif cls in GROUND1:
        emit("g", (0.5, 0.0))
    elif cls in RAIL1:
        emit("pwr", (0.5, 1.0))
    elif cls in PORT1:
        emit("1", (0.5, 0.5), 0.6)
    elif cls in BJT3:
        emit("c", (0.85, 0.0)); emit("b", (0.0, 0.5)); emit("e", (0.85, 1.0), 0.7)
    elif cls in FET3:
        emit("d", (0.85, 0.0)); emit("g", (0.0, 0.5)); emit("s", (0.85, 1.0), 0.7)
    elif cls in GATES:
        if cls == "not":
            emit("in0", (0.0, 0.5)); emit("out", (1.0, 0.5))
        else:
            emit("in0", (0.0, 0.3)); emit("in1", (0.0, 0.7)); emit("out", (1.0, 0.5))
    elif cls in OPAMP:
        emit("in-", (0.0, 0.3)); emit("in+", (0.0, 0.7)); emit("out", (1.0, 0.5))
    elif cls in AMP1:
        emit("in", (0.0, 0.5)); emit("out", (1.0, 0.5))
    elif cls in ICS:
        n_side = max(2, int(round(h / max(w, h) * 8)))
        for i in range(n_side):
            y = (i + 1) / (n_side + 1)
            emit(f"L{i}", (0.0, y), 0.5)
            emit(f"R{i}", (1.0, y), 0.5)
    elif cls in XFORMER:
        emit("p0", (0.0, 0.25)); emit("p1", (0.0, 0.75))
        emit("p2", (1.0, 0.25)); emit("p3", (1.0, 0.75), 0.7)
    elif cls in TGATE4:
        emit("a", (0.0, 0.3)); emit("b", (0.0, 0.7))
        emit("c", (1.0, 0.3)); emit("d", (1.0, 0.7), 0.6)
    elif cls in NONE:
        return []
    else:
        emit("p0", (0.0, 0.5), 0.5); emit("p1", (1.0, 0.5), 0.5)
    return out


def detect_edge_wires(bbox, dark, pad_out=10):
    """Scan narrow strips immediately outside the 4 bounding box edges.

    Returns dict mapping edge name ('left', 'right', 'top', 'bottom') to:
    (coord_x, coord_y, conf) where (coord_x, coord_y) is the exact 1D lead
    peak along that edge in absolute pixels, and conf is the dark-pixel ratio.
    """
    import numpy as np

    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h_img, w_img = dark.shape[:2]
    results = {}

    # Left edge: wire extending outward left (x < x0)
    lx0, lx1 = max(0, x0 - pad_out), max(0, x0)
    if lx1 > lx0 and y1 > y0:
        strip = dark[y0:y1, lx0:lx1]
        prof = strip.sum(axis=1)
        if prof.max() >= 2:
            results["left"] = (x0, y0 + int(np.argmax(prof)), float(prof.max() / (lx1 - lx0)))

    # Right edge: wire extending outward right (x > x1)
    rx0, rx1 = min(w_img, x1), min(w_img, x1 + pad_out)
    if rx1 > rx0 and y1 > y0:
        strip = dark[y0:y1, rx0:rx1]
        prof = strip.sum(axis=1)
        if prof.max() >= 2:
            results["right"] = (x1, y0 + int(np.argmax(prof)), float(prof.max() / (rx1 - rx0)))

    # Top edge: wire extending outward up (y < y0)
    ty0, ty1 = max(0, y0 - pad_out), min(h_img, y0)
    if ty1 > ty0 and x1 > x0:
        strip = dark[ty0:ty1, x0:x1]
        prof = strip.sum(axis=0)
        if prof.max() >= 2:
            results["top"] = (x0 + int(np.argmax(prof)), y0, float(prof.max() / (ty1 - ty0)))

    # Bottom edge: wire extending outward down (y > y1)
    by0, by1 = min(h_img, y1), min(h_img, y1 + pad_out)
    if by1 > by0 and x1 > x0:
        strip = dark[by0:by1, x0:x1]
        prof = strip.sum(axis=0)
        if prof.max() >= 2:
            results["bottom"] = (x0 + int(np.argmax(prof)), y1, float(prof.max() / (by1 - by0)))

    return results


def pins_for(cls, bbox, orientation=0, dark=None):
    """Return [(role, x, y, conf)] in pixels. Empty for non-pinned classes.

    If dark wire mask is provided, pins are snapped to the actual wire leads
    crossing the component boundaries.
    """
    raw = _raw_pins(cls, bbox, orientation)
    if not raw:
        return []

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    if dark is None:
        return [(r, *_at(bbox, pt), c) for r, pt, c in raw]

    wires = detect_edge_wires(bbox, dark)
    out = []

    if cls in AXIAL2 or cls in SOURCES2:
        ang, conf = pca_axis(bbox, dark)
        if ang is not None:
            off = min(abs(ang), abs(ang - 90), abs(ang - 180))
            if off > 20.0:
                exits = _axis_exits(bbox, ang)
                if exits is not None:
                    (ax, ay), (bx, by) = exits
                    return [("p0", float(ax), float(ay), 0.9),
                            ("p1", float(bx), float(by), 0.9)]
        o = orientation % 180
        if o == 0:
            # Horizontal: left (p0) and right (p1)
            yl = wires["left"][1] if "left" in wires else y0 + 0.5 * h
            yr = wires["right"][1] if "right" in wires else y0 + 0.5 * h
            out.append(("p0", x0, yl, 0.95))
            out.append(("p1", x1, yr, 0.95))
        else:
            # Vertical: top (p0) and bottom (p1)
            xt = wires["top"][0] if "top" in wires else x0 + 0.5 * w
            xb = wires["bottom"][0] if "bottom" in wires else x0 + 0.5 * w
            out.append(("p0", xt, y0, 0.95))
            out.append(("p1", xb, y1, 0.95))
        return out

    elif cls in BJT3 or cls in FET3:
        if orientation == 0:
            yb = wires["left"][1] if "left" in wires else y0 + 0.5 * h
            xc = wires["top"][0] if "top" in wires else x0 + 0.85 * w
            xe = wires["bottom"][0] if "bottom" in wires else x0 + 0.85 * w
            r_b, r_c, r_e = ("b", "c", "e") if cls in BJT3 else ("g", "d", "s")
            out.append((r_c, xc, y0, 0.95))
            out.append((r_b, x0, yb, 0.95))
            out.append((r_e, xe, y1, 0.95))
            return out

    elif cls in GROUND1:
        xg = wires["top"][0] if "top" in wires else x0 + 0.5 * w
        out.append(("g", xg, y0, 0.95))
        return out

    # Default fallback: canonical template coordinates
    for r, pt, c in raw:
        px, py = _at(bbox, pt)
        out.append((r, px, py, c))
    return out


def pca_axis(bbox, dark, inset=2):
    """Principal axis of dark pixels inside bbox (PCA).

    Returns (angle_deg 0-180 from +x, confidence) or (None, 0.0).
    Diagonal parts (R4/R9 zigzags) read ~45/135 deg here while the
    H-vs-V edge vote ties or picks a side; aspect ratio then strands
    one pin off the lead. Confidence = eigenvalue ratio (>=2.5) with
    >=40 dark pixels.
    """
    import numpy as np
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = dark.shape[:2]
    xa, xb = max(0, x0 + inset), min(w, x1 - inset)
    ya, yb = max(0, y0 + inset), min(h, y1 - inset)
    if xb <= xa + 2 or yb <= ya + 2:
        return None, 0.0
    sub = dark[ya:yb, xa:xb] > 0
    ys, xs = np.nonzero(sub)
    if len(xs) < 40:
        return None, 0.0
    xs = xs.astype(float)
    ys = ys.astype(float)
    C = np.cov(np.stack([xs, ys]))
    vals, vecs = np.linalg.eigh(C)
    if vals[1] <= 0:
        return None, 0.0
    ratio = vals[1] / max(vals[0], 1e-9)
    if ratio < 2.5:
        return None, 0.0
    vx, vy = vecs[0, 1], vecs[1, 1]
    ang = float(np.degrees(np.arctan2(vy, vx))) % 180.0
    return ang, float(min(ratio / 6.0, 3.0))


def _axis_exits(bbox, ang_deg):
    """Where the PCA axis through the box center exits the border."""
    import math
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    t = math.radians(ang_deg)
    dx, dy = math.cos(t), math.sin(t)
    cands = []
    if abs(dx) > 1e-9:
        for x, s in ((x0, -1), (x1, 1)):
            yy = cy + (x - cx) * dy / dx
            if y0 - 1 <= yy <= y1 + 1:
                cands.append((x, yy, s))
    if abs(dy) > 1e-9:
        for y, s in ((y0, -1), (y1, 1)):
            xx = cx + (y - cy) * dx / dy
            if x0 - 1 <= xx <= x1 + 1:
                cands.append((xx, y, s))
    if len(cands) < 2:
        return None
    # antipodal pair, deterministic: negative-axis end first (p0)
    along = [((px - cx) * dx + (py - cy) * dy, (px, py)) for px, py, _ in cands]
    along.sort()
    return (along[0][1], along[-1][1])


def score_orientation(cls, bbox, dark, orientations=(0, 90, 180, 270)):
    """Pick the rotation whose pins sit on the most wire pixels.

    Pins belong on wires; wrong orientations strand them in white space.
    Returns (best_orientation, best_score, margin) where margin indicates
    relative confidence over the alternative orientation.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Ground connects at top lead in schematics; do not rotate by aspect ratio.
    if cls in GROUND1:
        return 0, 1.0, float("inf")

    # Transistors: default to canonical vertical (0: base left, collector top, emitter bottom)
    if cls in BJT3 or cls in FET3:
        wires = detect_edge_wires(bbox, dark)
        has_vert_leads = ("top" in wires or "bottom" in wires)
        has_horiz_leads = ("left" in wires and "right" in wires) and not has_vert_leads
        if has_horiz_leads:
            return 90, 0.8, 2.0
        return 0, 0.9, 3.0

    # Axial 2-terminal components (resistors, caps, diodes, inductors, etc.)
    if cls in AXIAL2 or cls in SOURCES2:
        wires = detect_edge_wires(bbox, dark)
        h_score = (wires.get("left", (0, 0, 0.0))[2] + wires.get("right", (0, 0, 0.0))[2]) / 2.0
        v_score = (wires.get("top", (0, 0, 0.0))[2] + wires.get("bottom", (0, 0, 0.0))[2]) / 2.0

        if v_score > h_score + 0.15 and v_score >= 0.2:
            margin = v_score / (h_score + 1e-4)
            return 90, v_score, margin
        elif h_score > v_score + 0.15 and h_score >= 0.2:
            margin = h_score / (v_score + 1e-4)
            return 0, h_score, margin
        else:
            base = 0 if w >= h else 90
            return base, max(h_score, v_score), 1.0

    # General fallback using disc score
    scores = {}
    for o in orientations:
        raw = _raw_pins(cls, bbox, o)
        if not raw:
            continue
        scores[o] = sum(_disc_score(dark, *_at(bbox, pt)) for _, pt, _ in raw) / len(raw)
    if not scores:
        return orientations[0], 0.0, 1.0
    best = max(scores, key=lambda o: (scores[o], -o))
    # Exclude symmetric counterpart (180 deg) from runner-up
    rest = sorted((s for o, s in scores.items() if (o % 180) != (best % 180)), reverse=True)
    second = rest[0] if rest else 0.0
    best_s = scores[best]
    margin = best_s / second if second > 1e-9 else (float("inf") if best_s > 0 else 1.0)
    return best, best_s, margin


def ic_stub_pins(bbox, dark, band=8, min_support=4):
    """Find real wire stubs crossing an IC box border.

    Scans the perimeter band outside each edge for dark-pixel runs (= wires entering
    the package) and returns [(role, x, y, 0.8)]. Returns [] when fewer than
    2 stubs are found so the caller falls back to pitch estimation.
    """
    import numpy as np

    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = dark.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    out = []

    def groups_of(counts):
        groups, cur = [], []
        for i, v in enumerate(counts):
            if v >= min_support:
                if cur and i - cur[-1] > 3:
                    groups.append(cur)
                    cur = []
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return [int(sum(g) / len(g)) for g in groups]

    # Left edge: scan band outside x0
    if x0 - band >= 0 and y1 > y0:
        strip = dark[y0:y1, max(0, x0 - band):x0]
        for gi, c in enumerate(groups_of(strip.sum(axis=1))):
            out.append((f"L{gi}", x0, y0 + c, 0.8))

    # Right edge: scan band outside x1
    if x1 + band <= w and y1 > y0:
        strip = dark[y0:y1, x1:min(w, x1 + band)]
        for gi, c in enumerate(groups_of(strip.sum(axis=1))):
            out.append((f"R{gi}", x1, y0 + c, 0.8))

    # Top edge: scan band outside y0
    if y0 - band >= 0 and x1 > x0:
        strip = dark[max(0, y0 - band):y0, x0:x1]
        for gi, c in enumerate(groups_of(strip.sum(axis=0))):
            out.append((f"T{gi}", x0 + c, y0, 0.8))

    # Bottom edge: scan band outside y1
    if y1 + band <= h and x1 > x0:
        strip = dark[y1:min(h, y1 + band), x0:x1]
        for gi, c in enumerate(groups_of(strip.sum(axis=0))):
            out.append((f"B{gi}", x0 + c, y1, 0.8))

    return out if len(out) >= 2 else []
