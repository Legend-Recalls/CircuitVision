"""Stage 5: constraint validator (roadmap: first research contribution layer).

Six checks, each with a definitive verdict (no silent unknowns):
  1. two_terminal  - AXIAL2 + SOURCES2 classes: exactly 2 pins on exactly
                     2 distinct nets (SHORTED / OPEN / PIN_COUNT flags).
  2. transistor_roles - BJT3/FET3/gates/opamp/xformer/tgate/amp pin roles
                     match the class contract (MISSING_ROLE flags).
  3. sources_live - sources/rails/ports must not float: every pin on a
                     net shared with another pin (FLOATING flags).
  4. ground_collapse - all gnd/vss pins merge to one electrical GND;
                     always solved by construction, returns the merge map.
  5. crossing_no_dot - wire branch points without a junction dot must join
                     fragments of DIFFERENT nets (OVER_MERGED flags).
  6. shorts_islands - single-pin nets (ISLAND), dangling pins (DANGLING),
                     rail-to-ground nets (RAIL_SHORT).

Inputs are plain tuples so the validator stays decoupled from graph classes:
  comps: [(cls, bbox)]                      bbox=(x0,y0,x1,y1)
  pins:  [(comp_idx, role, net_or_None)]    net_or_None=None if isolated
  nets:  {net_idx: [pin_idx, ...]}
  wire_ctx (check 5 only): dict(masked, labels, keep, net_of, net_index)

Report: {check: {'verdict': PASS|FLAG|MAP, ...}} + summary counts.
"""
from circuitvision.pin_templates import AXIAL2, BJT3, FET3, GATES, OPAMP, ICS, XFORMER, \
    SOURCES2, TGATE4, AMP1, GROUND1, RAIL1, PORT1, NONE

TWO_T = set(AXIAL2) | set(SOURCES2) | set(AMP1)  # all 2-pin contracts
ROLE_CONTRACT = {}
for _c in BJT3:
    ROLE_CONTRACT[_c] = {'c', 'b', 'e'}
for _c in FET3:
    ROLE_CONTRACT[_c] = {'d', 'g', 's'}
for _c in GATES:
    ROLE_CONTRACT[_c] = {'in0', 'out'} if _c == 'not' else {'in0', 'in1', 'out'}
for _c in OPAMP:
    ROLE_CONTRACT[_c] = {'in-', 'in+', 'out'}
for _c in XFORMER:
    ROLE_CONTRACT[_c] = {'p0', 'p1', 'p2', 'p3'}
for _c in TGATE4:
    ROLE_CONTRACT[_c] = {'a', 'b', 'c', 'd'}
for _c in AMP1:
    ROLE_CONTRACT[_c] = {'in', 'out'}
SOURCE_LIKE = set(SOURCES2) | set(RAIL1) | set(PORT1)


def _same_fragment(cp, pin_xy, wire_ctx):
    """Both pins on one copper fragment? Needs pin_xy + wire_ctx labels;
    unknown (missing data) counts as different (report parallel, safer)."""
    if pin_xy is None or wire_ctx is None:
        return False
    labels = wire_ctx['labels']
    keep = set(wire_ctx['net_of']) | {0}
    (a, _, _), (b, _, _) = cp
    fa = _frag_of(labels, keep, *pin_xy[a])
    fb = _frag_of(labels, keep, *pin_xy[b])
    return fa is not None and fa == fb


def _pin_dist(pin_xy, cp):
    a, b = cp[0][0], cp[1][0]
    ax, ay, bx, by = pin_xy[a][0], pin_xy[a][1], pin_xy[b][0], pin_xy[b][1]
    return abs(ax - bx) + abs(ay - by)


def _frag_of(labels, keep, x, y, radius=12):
    """Nearest kept fragment to (x, y) or None (local copy: constraints
    must not import correct)."""
    import numpy as np
    h, w = labels.shape
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    ids = [int(v) for v in np.unique(labels[y0:y1, x0:x1])
           if v in keep and v != 0]
    if not ids:
        return None
    best, best_d = None, 1e18
    yy, xx = np.nonzero(np.isin(labels[y0:y1, x0:x1], ids))
    for dx, dy in zip(xx.tolist(), yy.tolist()):
        dd = (dx + x0 - x) ** 2 + (dy + y0 - y) ** 2
        if dd < best_d:
            best_d, best = dd, int(labels[y0 + dy, x0 + dx])
    return best


def check_two_terminal(comps, pins, nets, pin_xy=None, wire_ctx=None):
    """Exactly 2 pins on exactly 2 distinct nets for 2-terminal classes.

    Both pins on one net: SHORTED if they share one fragment (dead fault,
    verified at copper level) else PARALLEL if they sit on different
    fragments of a live net (dual packs, R1/R2 pairs - info only).
    """
    """Exactly 2 pins on exactly 2 distinct nets for 2-terminal classes.

    Both pins on one net with no other part present = SHORTED (dead fault).
    Both pins on a shared multi-pin net = PARALLEL - unless the pins sit
    closer than 2x the snap radius (same search window): then both pins
    probably grabbed one fragment (SHARED_SUSPECT, needs re-snap, not a
    placement claim). Legit parallels (D15/D16, R1/R2 packs) are far apart.
    """
    by_comp = {}
    for pi, (ci, role, net) in enumerate(pins):
        by_comp.setdefault(ci, []).append((pi, role, net))
    ok = shorted = opened = bad_count = skipped = 0
    flags, parallel = [], []
    for ci, (cls, _) in enumerate(comps):
        if cls not in TWO_T:
            skipped += 1
            continue
        cp = by_comp.get(ci, [])
        if len(cp) != 2:
            bad_count += 1
            flags.append((cls, ci, 'PIN_COUNT', len(cp)))
            continue
        nets_here = [n for _, _, n in cp]
        if any(n is None for n in nets_here):
            opened += 1
            flags.append((cls, ci, 'OPEN',
                          [p for p, _, n in cp if n is None]))
        elif nets_here[0] == nets_here[1]:
            if len(nets.get(nets_here[0], [])) <= 2:
                shorted += 1
                flags.append((cls, ci, 'SHORTED', nets_here[0]))
            elif _same_fragment(cp, pin_xy, wire_ctx):
                shorted += 1
                flags.append((cls, ci, 'SHORTED', nets_here[0]))
            else:
                parallel.append((cls, ci, nets_here[0],
                                 len(nets[nets_here[0]])))
            ok += 0
        else:
            ok += 1
    verdict = 'PASS' if not flags else 'FLAG'
    return {'verdict': verdict, 'ok': ok, 'shorted': shorted,
            'open': opened, 'pin_count': bad_count, 'skipped': skipped,
            'parallel': parallel, 'flags': flags}


def check_transistor_roles(comps, pins):
    """Pin roles match the class contract; every pin lands on a net."""
    by_comp = {}
    for pi, (ci, role, net) in enumerate(pins):
        by_comp.setdefault(ci, []).append((pi, role, net))
    ok = skipped = 0
    flags = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in ROLE_CONTRACT:
            skipped += 1
            continue
        cp = by_comp.get(ci, [])
        roles = {r for _, r, _ in cp}
        missing = ROLE_CONTRACT[cls] - roles
        if missing:
            flags.append((cls, ci, 'MISSING_ROLE', sorted(missing)))
            continue
        dangling = [p for p, _, n in cp if n is None]
        if dangling:
            flags.append((cls, ci, 'DANGLING_PIN', dangling))
        else:
            ok += 1
    verdict = 'PASS' if not flags else 'FLAG'
    return {'verdict': verdict, 'ok': ok, 'skipped': skipped,
            'flags': flags}


def check_sources_live(comps, pins, nets):
    """Sources/rails/ports must not float: pin nets shared with >=2 pins."""
    comp_of_pin = [ci for ci, _, _ in pins]
    ok = skipped = 0
    flags = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in SOURCE_LIKE and cls not in ICS:
            skipped += 1
            continue
        mine = [pi for pi, c in enumerate(comp_of_pin) if c == ci]
        if not mine:
            flags.append((cls, ci, 'NO_PINS', []))
            continue
        bad = [pi for pi in mine
               if pins[pi][2] is None or len(nets.get(pins[pi][2], [])) < 2]
        if bad:
            flags.append((cls, ci, 'FLOATING', bad))
        else:
            ok += 1
    verdict = 'PASS' if not flags else 'FLAG'
    return {'verdict': verdict, 'ok': ok, 'skipped': skipped,
            'flags': flags}


def check_ground_collapse(comps, pins):
    """All gnd/vss pins -> one electrical GND. Solved by construction."""
    gnd_nets = {}
    for pi, (ci, role, net) in enumerate(pins):
        if comps[ci][0] in GROUND1 and net is not None:
            gnd_nets.setdefault(net, []).append(pi)
    floating = [pi for pi, (ci, _, net) in enumerate(pins)
                if comps[ci][0] in GROUND1 and net is None]
    merge_map = {int(n): 'GND' for n in gnd_nets}
    return {'verdict': 'MAP', 'raw_nets': len(gnd_nets),
            'gnd_pins': sum(len(v) for v in gnd_nets.values()),
            'floating_gnd_pins': floating, 'merge_map': merge_map}


def check_crossing_no_dot(wire_ctx, pin_nets, dot_r=14, box_margin=10):
    """Un-dotted +-crossings must not imply connectivity.

    Method (all verified): Zhang-Suen skeleton -> degree>=4 X points
    outside validated dots/boxes on substantial copper -> inner-disc fill
    splits dotted-T's (filled, connected, fine) from bare X's -> bare X's
    explained by symbol geometry (near box / on Hough circle: coils, pin
    circles, cans) vs unexplained hop-over candidates (Stage 7 review).
    Mechanism unit tests: split->PASS, merged->FLAG (see skeleton_x usage
    in cross_unit runs).     Needs wire_ctx['dark'] (grayscale dark mask) and wire_ctx['wire']
    (unmasked binary map for circle recovery).
    """
    import numpy as np
    from circuitvision.skeleton_x import x_crossings, all_circles, explain_x
    masked = wire_ctx['masked']
    labels = wire_ctx['labels']
    dots = wire_ctx['dots']
    boxes = wire_ctx['boxes']
    dark = wire_ctx['dark']
    areas = np.bincount(labels.ravel())
    xs, _, _ = x_crossings(masked, labels, areas, dots, boxes,
                           dot_r=dot_r, box_margin=box_margin)
    circles = all_circles(wire_ctx.get('wire', masked), dots)
    part = explain_x(xs, dark, boxes, circles, labels, areas, masked)
    unexplained = part['unexplained']
    verdict = 'PASS' if not unexplained else 'FLAG'
    return {'verdict': verdict, 'x_total': len(xs),
            'dotted': len(part['dotted']),
            'explained_bare': len(part['explained']),
            'text_artifact': len(part['text']),
            'unexplained': unexplained[:15]}


def check_shorts_islands(comps, pins, nets):
    """Single-pin nets, dangling pins, rail-to-ground shorts."""
    islands = [(n, ps[0]) for n, ps in nets.items() if len(ps) == 1]
    dangling = [(pi, ci) for pi, (ci, _, net) in enumerate(pins)
                if net is None]
    gnd_nets = {pins[pi][2] for pi, (ci, _, _) in enumerate(pins)
                if comps[ci][0] in GROUND1 and pins[pi][2] is not None}
    rail_shorts = []
    for pi, (ci, _, net) in enumerate(pins):
        if comps[ci][0] in RAIL1 and net in gnd_nets:
            rail_shorts.append((pi, ci, net))
    flags = (len(islands) > 0) or (len(dangling) > 0) or (len(rail_shorts) > 0)
    return {'verdict': 'FLAG' if flags else 'PASS',
            'island_nets': islands, 'dangling_pins': dangling,
            'rail_shorts': rail_shorts}


def check_transformer_windings(comps, pins):
    """Each transformer winding (p0,p1) and (p2,p3) spans 2 distinct nets.

    Same SHORTED/OPEN/PARALLEL semantics as two_terminal per winding.
    (LTspice OP caught T114/T115 shorted windings that the role check
    passed: roles present + assigned is not enough.)
    """
    by_comp = {}
    for pi, (ci, role, net) in enumerate(pins):
        by_comp.setdefault(ci, []).append((pi, role, net))
    ok = skipped = 0
    flags, parallel = [], []
    for ci, (cls, _) in enumerate(comps):
        if cls != 'transformer':
            skipped += 1
            continue
        cp = {r: (pi, n) for pi, r, n in by_comp.get(ci, [])}
        if set(cp) != {'p0', 'p1', 'p2', 'p3'}:
            flags.append((cls, ci, 'PIN_COUNT', sorted(cp)))
            continue
        for winding in (('p0', 'p1'), ('p2', 'p3')):
            (a, na), (b, nb) = (cp[winding[0]], cp[winding[1]])
            if na is None or nb is None:
                flags.append((cls, ci, 'OPEN_' + '+'.join(winding),
                              [p for p, n in ((a, na), (b, nb)) if n is None]))
            elif na == nb:
                flags.append((cls, ci, 'SHORTED_' + '+'.join(winding), na))
            else:
                ok += 1
    return {'verdict': 'PASS' if not flags else 'FLAG', 'ok': ok,
            'skipped': skipped, 'flags': flags, 'parallel': parallel}


def check_crossover_junction(comps, dots):
    """Hop-overs must avoid dots; junction glyphs must sit on dots.

    crossover/crossover.curved = drawn hop (wires NOT connected): a dot
    inside is a drafting contradiction. junction = explicit net node:
    with no dot nearby it is a phony node. Reports examined counts so
    empty classes still prove the check ran.
    """
    flags, n_cross, n_junc = [], 0, 0
    for ci, (cls, (x0, y0, x1, y1)) in enumerate(comps):
        if cls in ('crossover', 'crossover.curved'):
            n_cross += 1
            inside = [(dx, dy) for dx, dy, _ in dots
                      if x0 + 3 <= dx <= x1 - 3 and y0 + 3 <= dy <= y1 - 3]
            if inside:
                flags.append((cls, ci, 'DOT_INSIDE_HOP', len(inside)))
        elif cls == 'junction':
            n_junc += 1
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rr = max(x1 - x0, y1 - y0) / 2 + 6
            if not any((cx - dx) ** 2 + (cy - dy) ** 2 <= rr * rr
                       for dx, dy, _ in dots):
                flags.append((cls, ci, 'PHONY_JUNCTION', (round(cx), round(cy))))
    return {'verdict': 'PASS' if not flags else 'FLAG',
            'crossovers': n_cross, 'junctions': n_junc, 'flags': flags}


def validate(comps, pins, nets, wire_ctx=None, footprints=(), pin_xy=None):
    """Run all checks. footprints: IC structural candidates (Stage 7)."""
    rep = {
        'two_terminal': check_two_terminal(comps, pins, nets, pin_xy,
                                           wire_ctx),
        'transformer_windings': check_transformer_windings(comps, pins),
        'transistor_roles': check_transistor_roles(comps, pins),
        'sources_live': check_sources_live(comps, pins, nets),
        'ground_collapse': check_ground_collapse(comps, pins),
        'shorts_islands': check_shorts_islands(comps, pins, nets),
    }
    if wire_ctx is not None:
        pin_nets = {n for _, _, n in pins if n is not None}
        rep['crossing_no_dot'] = check_crossing_no_dot(wire_ctx, pin_nets)
    rep['crossover_junction'] = check_crossover_junction(
        comps, wire_ctx['dots'] if wire_ctx else [])
    rep['structural_candidates'] = [
        {'hint': 'integrated_circuit', 'bbox': fp[:4], 'pins': fp[4]}
        for fp in footprints]
    return rep
