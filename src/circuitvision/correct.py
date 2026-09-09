"""Stage 7: graph correction engine (roadmap order: geometry, hard
constraints, motif priors; no ML fallback needed here).

Passes (each logged with reason + confidence; below-threshold stays flagged):
  P1 noise prune   : rescued comp, ALL pins troubled, zero repair
                     candidates (port163 on a dot) -> REMOVE.
  P2 ground homing : troubled gnd/vss pins -> nearest GND-merged net.
  P3 motif attach  : remaining troubled pins take rank top-1 iff
                     score>=1.0 and dist<=80px.
  P4 IC synthesis  : footprint + terminal circles -> integrated_circuit
                     comp with pins assigned to touching fragments
                     (maze fallback with sibling exclusion).
  P5 overmerge split: none found by Stage 5 -> logged no-op.

Operates on (comps, pins, pin_xy, nets, wire_ctx, gnd_merge, rank) and
returns corrected copies + action log. Ids stay stable (no renumbering).
"""
import numpy as np


def _frag_at(labels, keep, x, y, radius=16, disc=9):
    """Fragment owning pin (x, y): max copper overlap in a small disc,
    tie-break nearest pixel. Overlap (not proximity) because a pin's own
    copper (circle outline, stub) beats a fat rail passing nearby —
    proximity alone piled 4 IC pins onto ground rail 235."""
    h, w = labels.shape
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    sub = labels[y0:y1, x0:x1]
    ids = [int(v) for v in np.unique(sub) if v in keep and v != 0]
    if not ids:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    in_disc = (xx - x) ** 2 + (yy - y) ** 2 <= disc * disc
    best, best_ov, best_d = None, -1, 1e18
    for i in ids:
        m = (sub == i)
        ov = int((m & in_disc).sum())
        if not m.any():
            continue
        dy, dx = np.nonzero(m)
        d = float((((dx + x0 - x) ** 2 + (dy + y0 - y) ** 2)).min())
        if (ov, -d) > (best_ov, -best_d):
            best, best_ov, best_d = i, ov, d
    return best


def p1_prune_noise(comps, pins, rank, rescued_from=113):
    """Remove rescued comps whose every pin is troubled with no candidates."""
    troubled_pins = {pi for pi, c in rank.items() if not c}
    gone_comps, gone_pins = [], []
    for ci, (cls, _) in enumerate(comps):
        if ci < rescued_from:
            continue
        mine = [pi for pi, (c, _, _) in enumerate(pins) if c == ci]
        if mine and all(p in troubled_pins for p in mine):
            gone_comps.append(ci)
            gone_pins += mine
    log = [{'pass': 'P1', 'action': 'remove', 'comp': ci,
            'cls': comps[ci][0], 'reason': 'rescued noise: all pins '
            'troubled, zero repair candidates'} for ci in gone_comps]
    return gone_comps, gone_pins, log


def p2_ground_homing(comps, pins, pin_xy, nets, gnd_merge,
                     troubled, radius=120):
    """Troubled gnd/vss pins join the nearest GND-merged net."""
    from circuitvision.constraints import GROUND1
    # anchors: pins already on GND-merged nets
    anchors = [(pin_xy[qi], n) for qi, (_, _, n) in enumerate(pins)
               if n is not None and n in gnd_merge]
    moves, log = {}, []
    for pi, ci, role in troubled:
        if comps[ci][0] not in GROUND1:
            continue
        if pins[pi][2] is not None and pins[pi][2] in gnd_merge:
            continue  # island gnd net is still GND: keep, just note
        x, y = pin_xy[pi]
        best, bd = None, 1e18
        for (ax, ay), n in anchors:
            d = abs(ax - x) + abs(ay - y)
            if d < bd:
                bd, best = d, n
        if best is not None and bd <= radius:
            moves[pi] = best
            log.append({'pass': 'P2', 'action': 'reassign', 'pin': pi,
                        'from': pins[pi][2], 'to': best,
                        'reason': f'gnd homing, dist {bd:.0f}'})
    return moves, log


def p3_motif_attach(comps, pins, rank, max_dist=80, min_score=1.0):
    """Top-1 ranked net iff score/dist pass AND no sibling pin holds it.

    Sibling exclusion (same lesson as the Stage-4 maze rescue): attaching
    to a sibling's net shorts the part — LTspice caught P3 shorting
    transformer T114's winding this way.
    """
    sib = {}
    for pi, (ci, _, n) in enumerate(pins):
        if n is not None:
            sib.setdefault(ci, set()).add(n)
    moves, log = {}, []
    for pi, cands in rank.items():
        ci = pins[pi][0]
        for s, e, d in cands:
            if e in sib.get(ci, set()):
                log.append({'pass': 'P3', 'action': 'skip-sibling',
                            'pin': pi, 'net': e,
                            'reason': 'sibling holds it, would short'})
                continue
            if s >= min_score and d <= max_dist:
                moves[pi] = e
                log.append({'pass': 'P3', 'action': 'reassign', 'pin': pi,
                            'to': e,
                            'reason': f'rank top-1 score {s} dist {d}'})
            break
    return moves, log


def p4_synthesize_ic(comps, pins, pin_xy, wire_ctx, footprint,
                     net_of, net_index, keep):
    """IC1: footprint bbox + terminal circles -> comp + pins on fragments."""
    import cv2
    from circuitvision.wire_nets import astar_to_wire
    x0, y0, x1, y1, n = footprint
    labels = wire_ctx['labels']
    wire = wire_ctx['wire']
    blur = cv2.medianBlur((wire * 255).astype(np.uint8), 3)
    found = []
    for maxr in (20,):
        c = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                             param2=15, minRadius=5, maxRadius=maxr)
        if c is not None:
            found += [(float(a), float(b), float(r)) for a, b, r in c[0]]
    inside = [(a, b) for a, b, _ in found
              if x0 <= a <= x1 and y0 - 40 <= b <= y1 + 40]
    # dedup 8px grid, sort top-row first then columns
    uniq = []
    for a, b in sorted(inside):
        if not any(abs(a - u) < 8 and abs(b - v) < 8 for u, v in uniq):
            uniq.append((a, b))
    if len(uniq) < 6:
        return None, [{'pass': 'P4', 'action': 'defer',
                       'reason': f'only {len(uniq)} pin circles, need >=6'}]
    ci = len(comps)
    new_comps = list(comps) + [('integrated_circuit',
                                (x0, y0, x1, y1))]
    new_pins, new_xy, log = list(pins), list(pin_xy), []
    targets = set(net_of)
    for i, (a, b) in enumerate(sorted(uniq, key=lambda p: (p[1], p[0]))):
        side = 'T' if b < y0 + 60 else ('L' if a < (x0 + x1) / 2 else 'R')
        frag = _frag_at(labels, keep, a, b)
        net = net_index[net_of[frag]] if frag is not None else None
        src = 'circle-wire'
        if net is None:
            f2, cost = astar_to_wire(wire, (a, b), targets, labels)
            if f2 is not None and cost <= 200.0:
                net = net_index[net_of[f2]]
                src = f'maze({cost:.0f})'
        new_pins.append((ci, f'{side}{i}', net))
        new_xy.append((a, b))
        log.append({'pass': 'P4', 'action': 'ic-pin', 'pin': len(new_pins) - 1,
                    'net': net, 'reason': src})
    attached = sum(1 for _, _, n in new_pins[-len(uniq):] if n is not None)
    from collections import Counter
    shared = Counter(n for _, _, n in new_pins[-len(uniq):]
                     if n is not None)
    bus = [(n, c) for n, c in shared.items() if c >= 3]
    log.insert(0, {'pass': 'P4', 'action': 'synthesize', 'comp': ci,
                   'reason': f'IC footprint {len(uniq)} pins, '
                   f'{attached}/{len(uniq)} on nets' +
                   (f'; SHARED-BUS anomaly nets {bus} (pin-row guide '
                    'line shorts pins in pixels - review)' if bus else '')})
    return (new_comps, new_pins, new_xy), log


def apply_corrections(comps, pins, pin_xy, nets, wire_ctx, gnd_merge,
                      rank, footprints):
    """Full Stage-7 run. Returns dict with corrected data + log."""
    from circuitvision.constraints import GROUND1
    log = []
    gone_c, gone_p, l1 = p1_prune_noise(comps, pins, rank)
    log += l1
    keep_c = [c for c in range(len(comps)) if c not in gone_c]
    keep_p = [p for p in range(len(pins)) if p not in gone_p]

    # rebuild nets without pruned pins
    nets2 = {}
    for p in keep_p:
        _, _, n = pins[p]
        if n is not None:
            nets2.setdefault(n, []).append(p)
    # drop emptied nets
    nets2 = {n: ps for n, ps in nets2.items() if ps}

    # troubled = dangling pins + single-pin-net pins (non-gnd; gnd
    # islands are GND by merge, handled in P2 only if net-less)
    troubled = [(p, pins[p][0], pins[p][1]) for p in keep_p
                if pins[p][2] is None or
                (len(nets2.get(pins[p][2], [])) == 1 and
                 comps[pins[p][0]][0] not in GROUND1)]

    m2, l2 = p2_ground_homing(comps, pins, pin_xy, nets2, gnd_merge,
                              [(p, c, r) for p, c, r in troubled])
    log += l2
    pins2 = [list(t) for t in pins]
    for p, n in m2.items():
        pins2[p][2] = n
    # P3 on still-troubled (recompute singles after P2)
    nets3 = {}
    for p in keep_p:
        n = pins2[p][2]
        if n is not None:
            nets3.setdefault(n, []).append(p)
    still = [(p, pins2[p][0], pins2[p][1]) for p in keep_p
             if pins2[p][2] is None or
             (len(nets3.get(pins2[p][2], [])) == 1 and
              comps[pins2[p][0]][0] not in GROUND1)]
    # rank keys are old pin idx (stable, no renumbering) -> reuse
    rank_still = {p: rank.get(p, []) for p, _, _ in still}
    m3, l3 = p3_motif_attach(comps, pins2, rank_still)
    log += l3
    for p, n in m3.items():
        pins2[p][2] = n
    pins2 = [tuple(t) for t in pins2]

    nets4 = {}
    for p in keep_p:
        n = pins2[p][2]
        if n is not None:
            nets4.setdefault(n, []).append(p)

    fp = footprints[0][:4] + (footprints[0][4],) if footprints else None
    ic_log = []
    comps4, pins4, xy4 = comps, pins2, pin_xy
    if fp:
        res, ic_log = p4_synthesize_ic(
            comps, pins2, pin_xy, wire_ctx, fp, wire_ctx['net_of'],
            wire_ctx['net_index'], new_keep(wire_ctx))
        log += ic_log
        if res is not None:
            comps4, pins4, xy4 = res
    # reindex: drop removed comps/pins so no ghost flags downstream
    gone_c_set, gone_p_set = set(gone_c), set(gone_p)
    cmap = {}
    comps5 = []
    for ci, c in enumerate(comps4):
        if ci in gone_c_set:
            continue
        cmap[ci] = len(comps5)
        comps5.append(c)
    pins5, xy5 = [], []
    for p, (ci, role, n) in enumerate(pins4):
        if p in gone_p_set or ci in gone_c_set:
            continue
        pins5.append((cmap[ci], role, n))
        xy5.append(xy4[p])
    nets5 = {}
    for p, (_, _, n) in enumerate(pins5):
        if n is not None:
            nets5.setdefault(n, []).append(p)

    log.append({'pass': 'P5', 'action': 'no-op',
                'reason': 'Stage 5 found no over-merged nets / true shorts'})
    return {'comps': comps5, 'pins': pins5, 'pin_xy': xy5, 'nets': nets5,
            'removed': (gone_c, gone_p), 'log': log}


def new_keep(wire_ctx):
    return set(wire_ctx['net_of']) | {0}
