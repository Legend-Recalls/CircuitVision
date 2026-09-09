"""Graph -> LTspice .asc reconstruction (generic, pipeline-native).

Symbols placed from detection boxes (fixed LTspice sizes, R0/R90 only);
stubs join our pin coords (true lead positions) to symbol pins; every
pin gets a FLAG net label (our electrical net names, GND->0), so the
drawing is topologically exact with zero false-short risk from routed
copper. Verified by LTspice's own netlister (-netlist) against our
pin->net incidence.
Pin geometry (from installed lib/sym/*.asy + Hartly.asc R90=CCW proof):
  res/ind : R0 (16,16),(16,96)   R90 CCW (X-16,Y+16),(X-96,Y+16)
  cap/diode: R0 (16,0),(16,64)   R90 (X+0,Y+16),(X-64,Y+16)
  npn/pnp : R0 C(64,0) B(0,48) E(64,96)
  voltage : R0 (0,16),(0,96)     sw: same + NC(-48,80),(-48,32) unconnected
  IC/unknown-with-pins: RECTANGLE + stubs + FLAGs.
Refdes match netlist.py exactly (R<ci> C<ci> L<ci> D<ci> Q<ci> M<ci>
V<ci> I<ci> S<ci> L<ci>a/b K<ci> X<ci>) so the two artifacts cross-check.
"""
R2, C2 = {'resistor', 'lamp'}, {'capacitor'}


def _horiz(p0, p1):
    return abs(p0[0] - p1[0]) >= abs(p0[1] - p1[1])


def place(sym, x, y, rot):
    return sym, int(round(x)), int(round(y)), rot


def placements(comps, pins, pin_xy):
    """[(inst, sym, X, Y, rot, [(sp_idx_or_role, sx, sy)], value)] + notes."""
    by_comp = {}
    for pi, (ci, role, n) in enumerate(pins):
        by_comp.setdefault(ci, []).append((pi, role, n))
    out, notes = [], []
    for ci, (cls, bb) in enumerate(comps):
        P = {r: (pin_xy[pi], n) for pi, r, n in
             [(p, r, n) for p, (c, r, n) in enumerate(pins) if c == ci]}
        x0, y0, x1, y1 = bb
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if cls in R2 or cls in ('inductor', 'inductor.ferrite', 'fuse',
                                'switch', 'crystal', 'varistor', 'lamp',
                                'light', 'speaker', 'motor', 'antenna',
                                'thermistor', 'diac'):
            sym = 'sw' if cls == 'switch' else 'res'
            inst = f'S{ci}' if cls == 'switch' else f'R{ci}'
            val = 'SMOD' if cls == 'switch' else '1k'
            if 'p0' in P and 'p1' in P and _horiz(P['p0'][0], P['p1'][0]):
                X = max(P['p0'][0][0], P['p1'][0][0]) + 16
                Y = (P['p0'][0][1] + P['p1'][0][1]) / 2 - 16
                sp = [(1, X - 16, Y + 16), (2, X - 96, Y + 16)]
                out.append((inst, sym, X, Y, 'R90', sp, val))
            else:
                X = cx - (16 if sym == 'res' else 0)
                Y = min([P[r][0][1] for r in P] or [y0]) - 16
                ox = 16 if sym == 'res' else 0
                sp = [(1, X + ox, Y + 16), (2, X + ox, Y + 96)]
                out.append((inst, sym, X, Y, 'R0', sp, val))
        elif cls in C2 or cls in ('diode', 'diode.zener',
                                  'diode.light_emitting'):
            sym = 'cap' if cls in C2 else 'diode'
            inst = (f'C{ci}' if cls in C2 else f'D{ci}')
            val = '1u' if cls in C2 else 'DMOD'
            if 'p0' in P and 'p1' in P and _horiz(P['p0'][0], P['p1'][0]):
                X = max(P['p0'][0][0], P['p1'][0][0])
                Y = (P['p0'][0][1] + P['p1'][0][1]) / 2 - 16
                sp = [(1, X, Y + 16), (2, X - 64, Y + 16)]
                out.append((inst, sym, X, Y, 'R90', sp, val))
            else:
                X = cx - 16
                Y = min([P[r][0][1] for r in P] or [y0])
                sp = [(1, X + 16, Y), (2, X + 16, Y + 64)]
                out.append((inst, sym, X, Y, 'R0', sp, val))
        elif cls in ('npn', 'pnp', 'transistor.bjt', 'thyristor'):
            inst = f'Q{ci}'
            X = P.get('b', ((cx, cy), None))[0][0] if 'b' in P else cx - 32
            ys = []
            if 'c' in P:
                ys.append(P['c'][0][1])
            if 'e' in P:
                ys.append(P['e'][0][1] - 96)
            if 'b' in P:
                ys.append(P['b'][0][1] - 48)
            Y = sum(ys) / len(ys) if ys else cy - 48
            sp = [('c', X + 64, Y), ('b', X, Y + 48), ('e', X + 64, Y + 96)]
            out.append((inst, 'pnp' if cls == 'pnp' else 'npn', X, Y, 'R0',
                        sp, 'QN' if cls != 'pnp' else 'QP'))
        elif cls in ('mosfet', 'nmos', 'nmos.bulk', 'pmos', 'pmos.bulk'):
            inst = f'M{ci}'
            X = P.get('g', ((cx, cy), None))[0][0] if 'g' in P else cx - 32
            ys = []
            for r, off in (('d', 0), ('s', 96), ('g', 48)):
                if r in P:
                    ys.append(P[r][0][1] - off)
            Y = sum(ys) / len(ys) if ys else cy - 48
            sp = [('d', X + 64, Y), ('g', X, Y + 48), ('s', X + 64, Y + 96)]
            out.append((inst, 'nmos' if cls in ('mosfet', 'nmos', 'nmos.bulk')
                        else 'pmos', X, Y, 'R0', sp, 'MN'))
        elif cls == 'transformer' and set(P) == {'p0', 'p1', 'p2', 'p3'}:
            topy = min(P[r][0][1] for r in P) - 16
            la = (f'L{ci}a', 'ind', cx - 40, topy, 'R0',
                  [(1, cx - 24, topy + 16), (2, cx - 24, topy + 96)], '10u')
            lb = (f'L{ci}b', 'ind', cx + 8, topy, 'R0',
                  [(1, cx + 24, topy + 16), (2, cx + 24, topy + 96)], '10u')
            out.append(la)
            out.append(lb)
            notes.append(f'K{ci} L{ci}a L{ci}b 1 (add as TEXT directive)')
        elif cls in ('voltage', 'voltage.ac', 'voltage.dc',
                     'voltage.battery', 'voltage.dc.one_port',
                     'voltage.lines', 'current_source',
                     'current_source.ac', 'current_source.dc',
                     'dependent_current_source'):
            inst = (f'V{ci}' if 'voltage' in cls else f'I{ci}')
            X = cx
            Y = min([P[r][0][1] for r in P] or [y0]) - 16
            sp = [(1, X, Y + 16), (2, X, Y + 96)]
            out.append((inst, 'voltage', X, Y, 'R0', sp, '5'))
        elif cls == 'integrated_circuit':
            out.append((f'X{ci}', 'RECT', x0, y0, None,
                        [(r, pin_xy[pi][0], pin_xy[pi][1])
                         for pi, (c, r, n) in enumerate(pins) if c == ci],
                        'IC'))
        else:
            notes.append(f'{cls}#{ci}: no symbol (node-label only)')
    return out, notes


def _wire(x1, y1, x2, y2):
    # Skips degenerate zero-length segments: they hang the LTspice
    # netlister. Returns [line] or [].
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return [] if (x1, y1) == (x2, y2) else [f'WIRE {x1} {y1} {x2} {y2}']


def _stub(x1, y1, x2, y2):
    # Manhattan stub (horizontal then vertical): no diagonal scars.
    # Verification-safe: same endpoints, and -netlist re-checks anyway.
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    out = []
    if x1 != x2:
        out.append(f'WIRE {x1} {y1} {x2} {y1}')
    if y1 != y2:
        out.append(f'WIRE {x2} {y1} {x2} {y2}')
    return out


def ename(n):
    return '0' if n == 'GND' else f'N{n}'


def route_trees(pins, pin_xy, nets):
    """Manhattan MST per net over pin coords (viewing copper).

    May create crossings (false shorts in LTspice semantics) -> the
    caller MUST re-verify with -netlist and drop trees on mismatch.
    Returns [WIRE lines]. Dots unnecessary: LTspice joins T-junctions.
    """
    import math
    out = []
    for n, ps in nets.items():
        pts = [(int(round(pin_xy[p][0])), int(round(pin_xy[p][1])))
               for p in ps]
        pts = list(dict.fromkeys(pts))
        if len(pts) < 2:
            continue
        # Prim's, Manhattan distance
        used, fringe = [pts[0]], pts[1:]
        while fringe:
            bi, bd, bj = None, 1e18, None
            for i, a in enumerate(used):
                for j, b in enumerate(fringe):
                    dd = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    if dd < bd:
                        bi, bd, bj = a, dd, b
            ax, ay, bx, by = bi[0], bi[1], bj[0], bj[1]
            if ax != bx:
                out.append(f'WIRE {ax} {ay} {bx} {ay}')
            if ay != by:
                out.append(f'WIRE {bx} {ay} {bx} {by}')
            used.append(bj)
            fringe.remove(bj)
    return sorted(set(out))


def _snap_coords(pin_xy, tol=10, pin_nets=None):
    """Snap near-collinear pin coords to shared rails (median of cluster).

    Detection/pin noise puts rails ±5px off; unsnapped, MST routing
    stairsteps every segment. Clustering key includes the electrical net:
    same-net rails align, different nets can NEVER share coordinates
    (that would FLAG-merge them into a false short).
    pin_nets[i]: enet of pin i or None.
    """
    xs = sorted(p[0] for p in pin_xy)
    ys = sorted(p[1] for p in pin_xy)

    def snaps(indexed):
        # indexed: [(value, key)] clustered by value within key groups.
        by_key = {}
        for v, k in indexed:
            by_key.setdefault(k, []).append(v)
        med = {}
        for k, vs in by_key.items():
            vs = sorted(vs)
            clusters, cur = [], [vs[0]]
            for v in vs[1:]:
                if v - cur[-1] <= tol:
                    cur.append(v)
                else:
                    clusters.append(cur)
                    cur = [v]
            clusters.append(cur)
            for c in clusters:
                m = sorted(c)[len(c) // 2]
                for v in c:
                    med[(v, k)] = m
        return med

    mx = snaps([(p[0], pin_nets[i] if pin_nets else None)
                for i, p in enumerate(pin_xy)])
    my = snaps([(p[1], pin_nets[i] if pin_nets else None)
                for i, p in enumerate(pin_xy)])
    out = []
    for i, p in enumerate(pin_xy):
        k = pin_nets[i] if pin_nets else None
        out.append((mx[(p[0], k)], my[(p[1], k)]))
    return out


def gen_asc(comps, pins, pin_xy, nets, gnd_merge, path, title,
            trees=False, spread=1.6, snap_tol=10):
    en = lambda n: 'GND' if n in gnd_merge else n
    if snap_tol:
        pin_xy = _snap_coords(pin_xy, snap_tol,
                              [en(n) for _, _, n in pins])
    plc, notes = placements(comps, pins, pin_xy)
    # Detection boxes sit tighter than LTspice symbol footprints, so
    # symbols crowd. Expand all coordinates about the centroid: relative
    # layout (image-faithful) is preserved, clearance grows.
    # Verification-safe: only coords change, net names don't.
    pts = ([(X, Y) for _, _, X, Y, _, _, _ in plc] +
           [(sx, sy) for _, _, _, _, _, sp, _ in plc for _, sx, sy in sp] +
           [tuple(map(float, p)) for p in pin_xy])
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    S = lambda x, y: (cx + (x - cx) * spread, cy + (y - cy) * spread)
    if spread != 1.0:
        plc = [(i, s, *S(X, Y), r,
                [(k, *S(sx, sy)) for k, sx, sy in sp], v)
               for i, s, X, Y, r, sp, v in plc]
        pin_xy = [S(x, y) for x, y in pin_xy]
    en = lambda n: 'GND' if n in gnd_merge else n
    plc, notes = placements(comps, pins, pin_xy)
    by_comp = {}
    for pi, (ci, role, n) in enumerate(pins):
        by_comp.setdefault(ci, []).append((pi, role, n))
    L = [f'Version 4', f'SHEET 1 2000 1600']
    expected = {}  # (inst, sp) -> enet, for -netlist verification
    for inst, sym, X, Y, rot, sp, val in plc:
        if sym == 'RECT':
            ci = int(inst[1:])
            x0, y0, x1, y1 = [int(round(v)) for v in comps[ci][1]]
            (x0, y0), (x1, y1) = (tuple(map(int, S(x0, y0))),
                                  tuple(map(int, S(x1, y1))))
            L.append(f'RECTANGLE Normal {x0} {y0} {x1} {y1}')
            L.append(f'TEXT {x0} {y0 - 20} Left 2 {inst}({comps[ci][0]})')
            rorder = sorted(r for r, _, _ in sp)  # = subckt pin order
            for role, sx, sy in sp:
                pi = next(p for p, (c, r, n) in enumerate(pins)
                          if c == ci and r == role)
                n = pins[pi][2]
                if n is None:
                    continue
                L += _stub(sx, sy, pin_xy[pi][0], pin_xy[pi][1])
                # FLAG every our-pin: the only thing that guarantees each
                # our-pin copper is labeled even if trees fragment.
                L.append(f'FLAG {int(pin_xy[pi][0])} {int(pin_xy[pi][1])} '
                         f'{ename(en(n))}')
                expected[(inst, rorder.index(role))] = ename(en(n))
            continue
        L.append(f'SYMBOL {sym} {int(X)} {int(Y)} {rot}')
        L.append(f'SYMATTR InstName {inst}')
        L.append(f'SYMATTR Value {val}')
        ci = int(''.join(c for c in inst[1:] if c.isdigit()) or -1)
        # one-to-one: each symbol pin claims a DISTINCT our-pin (greedy by
        # distance). Nearest-each allowed two symbol pins to share one
        # our-pin, stranding the other (R3.2 read NC_01).
        cand = []
        for s in sp:
            sk, sx, sy = s
            for pi, role, n in by_comp.get(ci, []):
                if n is None:
                    continue
                dd = abs(pin_xy[pi][0] - sx) + abs(pin_xy[pi][1] - sy)
                cand.append((dd, sk, sx, sy, pi, role, n))
        cand.sort(key=lambda t: t[0])
        used_sp, used_pi = set(), set()
        for dd, sk, sx, sy, pi, role, n in cand:
            if sk in used_sp or pi in used_pi:
                continue
            used_sp.add(sk)
            used_pi.add(pi)
            L += _stub(sx, sy, pin_xy[pi][0], pin_xy[pi][1])
            L.append(f'FLAG {int(pin_xy[pi][0])} {int(pin_xy[pi][1])} '
                     f'{ename(en(n))}')
            expected[(inst, sk)] = ename(en(n))
    if trees:
        L += route_trees(pins, pin_xy, nets)
    maxy = 0
    for ln in L:
        p = ln.split()
        if p and p[0] in ('WIRE', 'FLAG') and len(p) >= 4:
            try:
                maxy = max(maxy, int(p[2]))
            except ValueError:
                pass
    L.append(f'TEXT 40 {maxy + 40} Left 2 {title} (CircuitVision)')
    for i, nt in enumerate(notes):
        L.append(f'TEXT 40 {maxy + 70 + 25 * i} Left 1 ;{nt}')
    open(path, 'w', newline='\n').write('\n'.join(L) + '\n')
    return expected


def verify_asc(path, expected):
    """LTspice -netlist parse vs expected (inst, sp) -> net incidence."""
    import subprocess
    exe = (r'C:\Users\wwwna\AppData\Local\Programs\ADI\LTspice\LTspice.exe')
    subprocess.run([exe, '-netlist', str(path)], capture_output=True,
                   timeout=120)
    nl = str(path)
    nl = nl[:nl.rfind('.')] + '.net'
    lines = open(nl).read().splitlines()
    got = {}
    for ln in lines:
        s = ln.strip()
        if not s or s[0] in '*.':
            continue
        t = s.split()
        if t[0][0] in 'RCLDVSI' and len(t) >= 4:
            inst = t[0]
            nn = 3 if inst[0] == 'Q' else (4 if inst[0] in 'MS' else 2)
            for i, tok in enumerate(t[1:1 + nn], start=1):
                got[(inst, i)] = tok
        elif t[0][0] == 'X' and len(t) >= 3:
            for i, tok in enumerate(t[1:-1]):
                got[(t[0], i)] = tok
    # role->order map for ours
    order = {'p0': 1, 'p1': 2, 'c': 1, 'b': 2, 'e': 3, 'd': 1, 'g': 2,
             's': 3, '+': 1, '-': 2, '1': 1}
    bad, checked = [], 0
    for (inst, sk), want in expected.items():
        key = (inst, order.get(sk, sk))
        if key not in got:
            bad.append((inst, sk, 'missing', want))
            continue
        checked += 1
        if got[key] != want and not (want == '0' and got[key] == '0'):
            bad.append((inst, sk, got[key], want))
    return {'checked': checked, 'mismatch': bad}
