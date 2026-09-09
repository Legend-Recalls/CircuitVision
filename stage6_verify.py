"""Stage 6 verification: motifs + repair ranking on the SMPS graph."""
import sys
sys.path.insert(0, '.')
from build_graph import build_graph
from constraints import validate
from motifs import all_motifs, rank_repairs, electrical_nets

out = build_graph(verbose=False)
comps, pins, nets = out['comps'], out['pins'], out['nets']
rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'])
gnd_merge = rep['ground_collapse']['merge_map']
mo = all_motifs(comps, pins, nets, gnd_merge)
print(f'graph: {len(comps)} comps, {len(pins)} pins, {len(nets)} pin-nets')
for name, inst in mo.items():
    print(f'[{name}]: {len(inst)}')
    for v in inst[:10]:
        w = dict(v)
        if 'tap_load' in w:
            w['tap_load'] = (len(v['tap_load']), v['tap_load'][:6])
        print(f'    {w}')

isl = rep['shorts_islands']
troubled = [(p, c, pins[p][1]) for p, c in isl['dangling_pins']]
troubled += [(p, pins[p][0], pins[p][1]) for _, p in isl['island_nets']]
enets = electrical_nets(pins, gnd_merge)
rank = rank_repairs(comps, pins, out['pin_xy'], enets, troubled,
                    wire_ctx=out['wire_ctx'], gnd_merge=gnd_merge)
print(f'[repair_ranking]: {len(rank)} troubled pins')
for pi, cands in rank.items():
    print(f'    pin{pi} {comps[pins[pi][0]][0]}{pins[pi][0]}.{pins[pi][1]} -> {cands}')
