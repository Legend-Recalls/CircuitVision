"""Stage 5 verification: full pipeline -> validate -> per-check verdicts."""
import sys
sys.path.insert(0, '.')
from build_graph import build_graph
from constraints import validate, TWO_T

out = build_graph()
comps, pins, nets = out['comps'], out['pins'], out['nets']
r = out['extract']
rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'],
               out['pin_xy'])

print(f'graph: {len(comps)} comps, {len(pins)} pins, {len(nets)} nets, '
      f"isolated={len(r['isolated'])} bridged={len(r['bridged'])}")
for name, v in rep.items():
    if name == 'structural_candidates':
        print(f'[{name}]: {v}')
        continue
    flags = v.get('flags', '')
    extra = {k: val for k, val in v.items() if k not in ('verdict', 'flags')}
    print(f"[{name}]: {v['verdict']} {extra}")
    if flags:
        for f in (flags if isinstance(flags, list) else [flags])[:12]:
            print(f'    FLAG {f}')
        if len(flags) > 12:
            print(f'    ... +{len(flags) - 12} more')

short_nets = {}
for ci, (cls, _) in enumerate(comps):
    if cls not in TWO_T:
        continue
    mine = [n for c2, _, n in pins if c2 == ci]
    if len(mine) == 2 and mine[0] is not None and mine[0] == mine[1]:
        short_nets.setdefault(mine[0], []).append((cls, ci))
print(f'shorted nets: {len(short_nets)}; sizes:',
      {n: (len(nets[n]), v[:3]) for n, v in list(short_nets.items())[:10]})
isl = rep['shorts_islands']
print('island owners:', [(comps[pins[p][0]][0], pins[p][0], pins[p][1])
                         for _, p in isl['island_nets']][:16])
print('dangling owners:', [(comps[c][0], c, pins[p][1])
                           for p, c in isl['dangling_pins']])
