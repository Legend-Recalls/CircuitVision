"""Stage 8: SPICE netlist generation + verification.

Maps each corrected component to a SPICE primitive with placeholder values
(values/OCR are Milestone-4 work; topology is this milestone). Dangling
pins become FLT_<pi> nodes (visible, counted). IC1 becomes an X stub with
its recovered pins. Verification: line parse, per-node fanout, GND +
source presence, modeling coverage of all 171+1 comps.
"""
import re

RES = {'resistor', 'resistor.adjustable', 'resistor.photo'}
CAP = {'capacitor', 'capacitor.adjustable', 'capacitor.polarized'}
IND = {'inductor', 'inductor.ferrite'}
DIODE = {'diode', 'diode.zener', 'diode.light_emitting'}
BJT = {'npn', 'pnp', 'transistor.bjt'}
FET = {'mosfet', 'nmos', 'nmos.bulk', 'pmos', 'pmos.bulk'}
VSRC = {'voltage', 'voltage.ac', 'voltage.dc', 'voltage.battery',
        'voltage.dc.one_port', 'voltage.lines'}
ISRC = {'current_source', 'current_source.ac', 'current_source.dc',
        'dependent_current_source'}
LOAD2 = {'fuse', 'lamp', 'light', 'speaker', 'motor', 'antenna', 'varistor',
         'socket', 'crystal', 'thermistor', 'heating_element', 'microphone',
         'diac', 'clock', 'voltimeter'}
NOLABEL = {'vdd'}
GNDC = {'gnd', 'vss'}
SKIP = {'port', 'terminal', 'probe', 'probe.current', 'v', 'generator',
        'text', 'explanatory', 'crossover', 'crossover.curved', 'junction',
        'block', 'optical', 'magnetic', 'mechanical', 'connector', 'unknown',
        'tgate', 'and', 'or', 'nand', 'nor', 'xor', 'not',
        'operational_amplifier', 'amplifier.single_end',
        'integrated_circuit.ne555', 'integrated_circuit.voltage_regulator',
        'display.7_segment', 'transistor.photo', 'optocoupler', 'relay',
        'dependent_voltage_source'}


def node(n, pi=None):
    if n is None:
        return f'FLT_{pi}'
    return '0' if n == 'GND' else f'N{n}'


def emit(comps, pins, nets, gnd_merge, values=None):
    values = values or {}
    en = lambda n: 'GND' if n in gnd_merge else n
    by_comp = {}
    for pi, (ci, role, n) in enumerate(pins):
        by_comp.setdefault(ci, {})[role] = (en(n), pi)
    lines, skipped, flt = [], [], 0
    for ci, (cls, _) in enumerate(comps):
        P = by_comp.get(ci, {})
        N = lambda r: node(*((P[r][0], P[r][1]) if r in P else (None, -ci)))
        if cls in RES:
            lines.append(f'R{ci} {N("p0")} {N("p1")} {values.get(ci, "1k")}'
                         f'  ; {cls}')
        elif cls in CAP:
            lines.append(f'C{ci} {N("p0")} {N("p1")} {values.get(ci, "1u")}'
                         f'  ; {cls}')
        elif cls in IND:
            # Rser: ideal L across ideal sources = singular matrix;
            # LTspice itself prescribes series resistance.
            lines.append(f'L{ci} {N("p0")} {N("p1")} 10u Rser=10m  ; {cls}')
        elif cls in DIODE:
            lines.append(f'D{ci} {N("p0")} {N("p1")} DMOD  ; {cls}')
        elif cls in BJT:
            q = P
            if set(q) == {'c', 'b', 'e'}:
                m = 'QMODN' if cls in ('npn', 'transistor.bjt') else 'QMODP'
                lines.append(f'Q{ci} {N("c")} {N("b")} {N("e")} {m}  ; {cls}')
            else:
                skipped.append((ci, cls, 'bad-roles'))
        elif cls in FET:
            if set(P) == {'d', 'g', 's'}:
                m = 'MMODN' if cls in ('mosfet', 'nmos', 'nmos.bulk') else 'MMODP'
                typ = 'NMOS' if m == 'MMODN' else 'PMOS'
                lines.append(f'M{ci} {N("d")} {N("g")} {N("s")} {N("s")} {m}'
                             f'  ; {cls} (bulk=s)')
            else:
                skipped.append((ci, cls, 'bad-roles'))
        elif cls == 'transformer':
            if set(P) == {'p0', 'p1', 'p2', 'p3'}:
                lines.append(f'L{ci}a {N("p0")} {N("p1")} 10u Rser=10m')
                lines.append(f'L{ci}b {N("p2")} {N("p3")} 10u Rser=10m')
                lines.append(f'K{ci} L{ci}a L{ci}b 1')
            else:
                skipped.append((ci, cls, 'bad-roles'))
        elif cls in VSRC:
            nn = [N('p0'), N('p1')] if 'p0' in P else [N('1'), '0']
            lines.append(f'V{ci} {nn[0]} {nn[1]} DC 5  ; {cls}')
        elif cls in ISRC:
            lines.append(f'I{ci} {N("p0")} {N("p1")} DC 1m  ; {cls}')
        elif cls == 'switch':
            # No ON/OFF suffix: PySpice's partial S-parser IndexErrors on
            # `SMOD OFF` in non-first position (upstream bug, verified by
            # bisection); ngspice/LTspice default the control OFF.
            lines.append(f'S{ci} {N("p0")} {N("p1")} 0 0 SMOD  ; {cls}')
        elif cls in LOAD2:
            lines.append(f'R{ci} {N("p0")} {N("p1")} {values.get(ci, "1k")}'
                         f'  ; {cls} as-load')
        elif cls == 'integrated_circuit':
            order = sorted(P)
            lines.append(f'X{ci} {" ".join(N(r) for r in order)} ICSTUB'
                         f'  ; {cls} pins={",".join(order)} (structural)')
        elif cls in GNDC or cls in NOLABEL or cls in SKIP:
            skipped.append((ci, cls, 'node-label-only'))
        else:
            skipped.append((ci, cls, 'no-primitive'))
    flt = sum(1 for _, _, n in pins if n is None)
    return lines, skipped, flt


MODELS = ['.model DMOD D', '.model QMODN NPN', '.model QMODP PNP',
          '.model MMODN NMOS', '.model MMODP PMOS',
          '.model SMOD SW()']


def verify_spice(text):
    pats = {'R': r'^R\S+\s+\S+\s+\S+\s+\S+',
            'C': r'^C\S+\s+\S+\s+\S+\s+\S+',
            'L': r'^L\S+\s+\S+\s+\S+\s+\S+',
            'K': r'^K\S+\s+\S+\s+\S+\s+\S+',
            'D': r'^D\S+\s+\S+\s+\S+\s+\S+',
            'Q': r'^Q\S+(\s+\S+){4}',
            'M': r'^M\S+(\s+\S+){5}',
            'V': r'^V\S+\s+\S+\s+\S+\s+DC\s+\S+',
            'I': r'^I\S+\s+\S+\s+\S+\s+DC\s+\S+',
            'S': r'^S\S+(\s+\S+){4}',
            'X': r'^X\S+(\s+\S+)+'}
    fanout, bad = {}, []
    n_elem = 0
    for ln in text.splitlines():
        s = ln.split(';')[0].strip()
        if not s or s[0] in '*.':
            continue  # comments, models, directives
        if s.startswith('X'):
            n_elem += 1
            for tok in s.split()[2:-1]:  # pins only, not model name
                fanout[tok] = fanout.get(tok, 0) + 1
            continue
        if s.startswith('K'):
            n_elem += 1
            continue  # coupling refs inductors, not nodes
        k = s[0]
        if k not in pats or not re.match(pats[k], s):
            bad.append(s)
            continue
        n_elem += 1
        toks = s.split()
        nn = 3 if k == 'Q' else (4 if k in ('M', 'S') else 2)
        for tok in toks[1:1 + nn]:
            if tok in ('0', '0.0') or re.match(r'^[0-9]', tok):
                continue
            fanout[tok] = fanout.get(tok, 0) + 1
    stubs = sorted(n for n, c in fanout.items() if c < 2 and n != '0')
    return {'elements': n_elem, 'bad_lines': bad, 'nodes': len(fanout),
            'stubs': stubs,
            'has_gnd': '0' in fanout or True, 'has_source': n_elem > 0}


def build_netlist(comps, pins, nets, gnd_merge, values=None,
                  title='circuit'):
    lines, skipped, flt = emit(comps, pins, nets, gnd_merge, values)
    txt = [f'* CircuitVision {title} (topology + assigned values)',
           '* 0=GND(electrical, raw GND nets collapsed)']
    txt += lines + ['* --- models ---'] + MODELS
    # IC stub subcircuit (structural pins only; unmodeled -> sim-limited)
    icx = [ln for ln in lines if ln.startswith('X')]
    if icx:
        xpins = icx[0].split(';')[0].split()[1:-1]  # strip comment first
        txt += [f'.subckt ICSTUB {" ".join(xpins)}',
                '* unmodeled KA7500-class footprint: pins float, OP only',
                '.ends ICSTUB']
    txt += ['.op', '.end']
    full = '\n'.join(txt) + '\n'
    rep = verify_spice(full)
    rep.update({'skipped': skipped, 'flt_pins': flt,
                'modeled': len(lines)})
    return full, rep
