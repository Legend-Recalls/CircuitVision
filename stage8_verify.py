"""Stage 8 verification: correct -> netlist -> report."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from build_graph import build_graph
from constraints import validate
from motifs import rank_repairs, electrical_nets
from correct import apply_corrections
from netlist import build_netlist

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
c2, p2, n2 = fix['comps'], fix['pins'], fix['nets']
rep2 = validate(c2, p2, n2, out['wire_ctx'], [])
gm2 = rep2['ground_collapse']['merge_map']
txt, vr = build_netlist(c2, p2, n2, gm2)
with open('demo/outputs/smps.sp', 'w', newline='\n') as f:
    f.write(txt)
print(f'netlist: {vr["modeled"]} elements, {vr["nodes"]} nodes, '
      f'flt={vr["flt_pins"]}, bad={len(vr["bad_lines"])}')
for b in vr['bad_lines'][:10]:
    print('  BAD:', b)
print('stubs (<2 conn):', vr['stubs'][:15], f'+{max(0, len(vr["stubs"]) - 15)} more'
      if len(vr['stubs']) > 15 else '')
print('skipped:', vr['skipped'][:20])
print('saved demo/outputs/smps.sp')
n_skip_model = sum(1 for _, c, _ in vr['skipped'] if not c.startswith('_model'))
n_xfmr = sum(1 for c, _ in c2 if c == 'transformer')
modeled_comps = vr['modeled'] - 2 * n_xfmr  # transformers emit L,L,K
print(f'coverage: modeled-comps={modeled_comps} skipped-nodes={n_skip_model} '
      f'total-comps={len(c2)} match={modeled_comps + n_skip_model == len(c2)}')
