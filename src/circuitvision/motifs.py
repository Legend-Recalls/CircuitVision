"""Stage 6: semantic motif reasoning on the recovered graph (not pixels).

Electrical nets = raw nets with the Stage-5 ground map applied (27 raw
GND nets -> one GND; without this no divider-to-ground ever matches).
Matchers are structural (class + topology), each returning instances with
member components so a human can verify them on the sheet:
  divider, decoupling, rc_lowpass, rc_highpass, common_emitter,
  emitter_follower, parallel_pair.
Plus rank_repairs(): motif-prioritized candidate nets for every
dangling/island pin (Stage 7 consumes the ranking, not raw pixels).
"""
RES = {'resistor', 'resistor.adjustable', 'resistor.photo'}
CAP = {'capacitor', 'capacitor.adjustable', 'capacitor.polarized'}
DIODE = {'diode', 'diode.zener', 'diode.light_emitting'}
BJT = {'npn', 'pnp', 'transistor.bjt'}
GND1 = {'gnd', 'vss'}


def electrical_nets(pins, gnd_merge):
    """pin_idx -> enet ('GND' or raw net). None stays None."""
    return [None if n is None else gnd_merge.get(n, n)
            for _, _, n in pins]


def _by_comp(pins):
    out = {}
    for pi, (ci, role, _) in enumerate(pins):
        out.setdefault(ci, []).append((pi, role))
    return out


def _two_ends(ci, comps, pins, enets):
    """(enet_a, enet_b) or None if not exactly 2 live pins."""
    mine = [(pi, r) for pi, (c, r, _) in enumerate(pins) if c == ci]
    if len(mine) != 2:
        return None
    e = [enets[pi] for pi, _ in mine]
    if any(v is None for v in e):
        return None
    return tuple(e)


def _pin_nets(enets):
    nets = {}
    for pi, e in enumerate(enets):
        if e is not None:
            nets.setdefault(e, []).append(pi)
    return nets


def _roles_of(ci, pins):
    return {r: pi for pi, (c, r, _) in enumerate(pins) if c == ci}


def find_dividers(comps, pins, enets):
    """R-R series across 3 distinct enets with a loaded tap."""
    pin_nets = _pin_nets(enets)
    res = [(ci, _two_ends(ci, comps, pins, enets))
           for ci, (cls, _) in enumerate(comps) if cls in RES]
    res = [(ci, e) for ci, e in res if e and e[0] != e[1]]
    out, seen = [], set()
    for i, (a, (a1, a2)) in enumerate(res):
        for b, (b1, b2) in res[i + 1:]:
            shared = set((a1, a2)) & set((b1, b2))
            if len(shared) != 1:
                continue
            tap = next(iter(shared))
            if tap == 'GND':
                continue  # two pull-downs, not a divider; GND fanout
            # would fake hundreds of instances (measured 163 -> handful).
            if len(pin_nets.get(tap, [])) > 5:
                continue  # tap is a rail (e.g. decoupling rail 350),
            # not an intermediate node: combinatorial R pairs, not dividers.
            ends = {a1, a2, b1, b2} - {tap}
            if len(ends) != 2:
                continue
            key = tuple(sorted((a, b)) + [tap])
            if key in seen:
                continue
            seen.add(key)
            tap_pins = pin_nets.get(tap, [])
            load = [(pins[p][0], pins[p][1]) for p in tap_pins
                    if pins[p][0] not in (a, b)]
            out.append({'r_top': a, 'r_bot': b, 'tap': tap,
                        'ends': sorted(ends, key=str),
                        'tap_load': load})
    return out


def find_decoupling(comps, pins, enets):
    """C across (rail, GND) with the rail fanning out."""
    pin_nets = _pin_nets(enets)
    out = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in CAP:
            continue
        e = _two_ends(ci, comps, pins, enets)
        if not e or e[0] == e[1]:
            continue
        if 'GND' not in e:
            continue
        rail = e[1] if e[0] == 'GND' else e[0]
        if len(pin_nets.get(rail, [])) >= 3:
            out.append({'cap': ci, 'rail': rail,
                        'rail_pins': len(pin_nets[rail])})
    return out


def find_rc(comps, pins, enets, kind='lowpass'):
    """lowpass: R(a-b) + C(b-GND). highpass: C(a-b) + R(b-GND).

    Node must be small (<=5 pins): rail hubs with a decoupling cap
    fake one filter per resistor (measured: rail 350 caused 6x).
    """
    first, second = (RES, CAP) if kind == 'lowpass' else (CAP, RES)
    out = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in first:
            continue
        e = _two_ends(ci, comps, pins, enets)
        if not e or e[0] == e[1] or 'GND' in e:
            continue
        for cj, (cls2, _) in enumerate(comps):
            if cls2 not in second or cj == ci:
                continue
            f = _two_ends(cj, comps, pins, enets)
            if not f or len(set(f)) != 2:
                continue
            shared = set(e) & set(f)
            if len(shared) != 1:
                continue
            node = next(iter(shared))
            if node == 'GND':
                continue
            if len(_pin_nets(enets).get(node, [])) > 5:
                continue  # rail hub, not a filter node
            other = [v for v in f if v != node][0]
            if other == 'GND':
                out.append({'series': ci, 'shunt': cj, 'node': node})
    return out


def find_common_emitter(comps, pins, enets):
    """c -> R -> big net; e -> GND (direct or via R); b driven."""
    pin_nets = _pin_nets(enets)
    out = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in BJT:
            continue
        roles = _roles_of(ci, pins)
        if set(roles) != {'c', 'b', 'e'}:
            continue
        ec, eb, ee = (enets[roles[r]] for r in ('c', 'b', 'e'))
        if None in (ec, eb, ee):
            continue
        # collector through a resistor to a fat net (rail)
        pull = None
        for o, (cl, _) in enumerate(comps):
            if cl not in RES:
                continue
            f = _two_ends(o, comps, pins, enets)
            if f and ec in f:
                w = f[1] if f[0] == ec else f[0]
                if len(pin_nets.get(w, [])) >= 4:
                    pull = (o, w)
                    break
        if pull is None:
            continue
        # emitter to GND directly or through a resistor
        e_gnd = (ee == 'GND')
        er = None
        if not e_gnd:
            for o, (cl, _) in enumerate(comps):
                if cl not in RES:
                    continue
                f = _two_ends(o, comps, pins, enets)
                if f and ee in f and 'GND' in f:
                    er = o
                    break
        if not (e_gnd or er is not None):
            continue
        out.append({'q': ci, 'cls': cls, 'collector_r': pull[0],
                    'rail': pull[1], 'emitter_r': er,
                    'base_net': eb,
                    'base_driven': len(pin_nets.get(eb, [])) > 1})
    return out


def find_emitter_follower(comps, pins, enets):
    """c -> fat net (rail); e -> R -> GND; b driven."""
    pin_nets = _pin_nets(enets)
    out = []
    for ci, (cls, _) in enumerate(comps):
        if cls not in BJT:
            continue
        roles = _roles_of(ci, pins)
        if set(roles) != {'c', 'b', 'e'}:
            continue
        ec, eb, ee = (enets[roles[r]] for r in ('c', 'b', 'e'))
        if None in (ec, eb, ee) or ee == 'GND':
            continue
        if len(pin_nets.get(ec, [])) < 4:
            continue
        er = None
        for o, (cl, _) in enumerate(comps):
            if cl not in RES:
                continue
            f = _two_ends(o, comps, pins, enets)
            if f and ee in f and 'GND' in f:
                er = o
                break
        if er is None:
            continue
        out.append({'q': ci, 'cls': cls, 'emitter_r': er,
                    'rail': ec, 'base_net': eb,
                    'base_driven': len(pin_nets.get(eb, [])) > 1})
    return out


def find_parallel_pair(comps, pins, enets):
    """Two diodes on the same unordered enet pair (dual-diode pack,
    D15/D16-style). Polarity (anti- vs same-direction) needs a/k
    resolution the templates don't provide, so this claims parallel
    connection only — verified visually for [66,169]."""
    ends = {}
    for ci, (cls, _) in enumerate(comps):
        if cls not in DIODE:
            continue
        e = _two_ends(ci, comps, pins, enets)
        if not e or e[0] == e[1]:
            continue
        ends.setdefault(tuple(sorted(e, key=str)), []).append(ci)
    return [{'diodes': v, 'nets': list(k)} for k, v in ends.items()
            if len(v) >= 2]


def all_motifs(comps, pins, nets, gnd_merge):
    enets = electrical_nets(pins, gnd_merge)
    return {'voltage_divider': find_dividers(comps, pins, enets),
            'decoupling': find_decoupling(comps, pins, enets),
            'rc_lowpass': find_rc(comps, pins, enets, 'lowpass'),
            'rc_highpass': find_rc(comps, pins, enets, 'highpass'),
            'common_emitter': find_common_emitter(comps, pins, enets),
            'emitter_follower': find_emitter_follower(comps, pins, enets),
            'parallel_pair': find_parallel_pair(comps, pins, enets)}


def _copper_index(wire_ctx, gnd_merge):
    """Closure: copper_dist(x, y, enet, radius) via labeled fragments in
    a window around (x, y). Maps fragment -> raw net -> electrical net."""
    import numpy as np
    labels = wire_ctx['labels']
    net_of, net_index = wire_ctx['net_of'], wire_ctx['net_index']
    keep = set(net_of) | {0}
    gm = gnd_merge or {}

    def copper_dist(x, y, enet, radius):
        h, w = labels.shape
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        sub = labels[y0:y1, x0:x1]
        yy, xx = np.nonzero(sub)
        best = float('inf')
        for dx, dy in zip(xx.tolist(), yy.tolist()):
            f = int(sub[dy, dx])
            if f not in keep or f == 0:
                continue
            e = gm.get(net_index[net_of[f]], net_index[net_of[f]])
            if e != enet:
                continue
            d = abs(dx + x0 - x) + abs(dy + y0 - y)
            if d < best:
                best = d
                if best == 0:
                    break
        return best

    return copper_dist


def rank_repairs(comps, pins, pin_xy, enets, troubled, radius=120,
                 wire_ctx=None, gnd_merge=None):
    """Candidate live nets for each troubled pin, motif-prioritized.

    troubled: [(pin_idx, comp_idx, role)]. Score: +2 divider-tap or
    fat-rail (>=4 pins) target, +2 GND target for gnd-class pins,
    +1 live (>=3 pins); distance penalty per 40px. Distance is to the
    nearest net COPPER (fragment pixels in a window), not just the
    nearest pin: pins overstate gaps on long rails (cir1 R9: true bus
    copper 15px away, nearest pin 103px). Returns
    {pin_idx: [(score, enet, dist), ...]} top-3.
    """
    from circuitvision.constraints import GROUND1
    pin_nets = _pin_nets(enets)
    taps = set()
    for d in find_dividers(comps, pins, enets):
        taps.add(d['tap'])
    fat = {e for e, ps in pin_nets.items() if len(ps) >= 4}
    live = {e for e, ps in pin_nets.items() if len(ps) >= 3}
    copper = _copper_index(wire_ctx, gnd_merge) if wire_ctx is not None else None
    out = {}
    for pi, ci, role in troubled:
        x, y = pin_xy[pi]
        own = enets[pi]
        cands = {}
        for qi, e in enumerate(enets):
            if e is None or qi == pi or e == own:
                continue  # never suggest the pin's own (island) net
            d = abs(pin_xy[qi][0] - x) + abs(pin_xy[qi][1] - y)
            if copper is not None:
                d = min(d, copper(x, y, e, radius))
            if d > radius:
                continue
            s = 0
            if e in taps:
                s += 2
            if e in fat:
                s += 2
            elif e in live:
                s += 1
            if e == 'GND' and comps[ci][0] in GROUND1:
                s += 2
            s -= d / 40.0
            if e not in cands or s > cands[e][0]:
                cands[e] = (round(s, 2), round(d))
        out[pi] = sorted(((s, e, d) for e, (s, d) in cands.items()),
                         reverse=True)[:3]
    return out
