"""Stage 7 verification: correct -> re-validate -> before/after."""
import sys
sys.path.insert(0, '.')
from build_graph import build_graph
from constraints import validate
from motifs import all_motifs, rank_repairs, electrical_nets
from correct import apply_corrections

out = build_graph(verbose=False)
comps, pins, nets = out['comps'], out['pins'], out['nets']
rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'])
gm = rep['ground_collapse']['merge_map']
enets = electrical_nets(pins, gm)
isl = rep['shorts_islands']
troubled = [(p, c, pins[p][1]) for p, c in isl['dangling_pins']]
troubled += [(p, pins[p][0], pins[p][1]) for _, p in isl['island_nets']]
rank = rank_repairs(comps, pins, out['pin_xy'], enets, troubled,
                    wire_ctx=out['wire_ctx'], gnd_merge=gm)

fix = apply_corrections(comps, pins, out['pin_xy'], nets, out['wire_ctx'],
                        gm, rank, out['footprints'])
print('--- correction log ---')
for a in fix['log']:
    print(' ', a)
c2, p2, n2 = fix['comps'], fix['pins'], fix['nets']
print(f'before: {len(comps)} comps {len(pins)} pins {len(nets)} nets')
print(f'after:  {len(c2)} comps {len(p2)} pins {len(n2)} nets '
      f'(removed comps {fix["removed"][0]})')
rep2 = validate(c2, p2, n2, out['wire_ctx'], [])
for name, v in rep2.items():
    if name == 'structural_candidates':
        continue
    fl = v.get('flags', '')
    nfl = len(fl) if isinstance(fl, list) else 0
    print(f'[{name}]: {v["verdict"]} flags={nfl} '
          f'{ {k: val for k, val in v.items() if k not in ("verdict", "flags")} }')
    if isinstance(fl, list):
        for f in fl[:10]:
            print('    FLAG', f)
mo2 = all_motifs(c2, p2, n2, rep2['ground_collapse']['merge_map'])
print({k: len(v) for k, v in mo2.items()})
