"""Render current graph: net overlay, motif overlay, subgraph diagram."""
import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from circuitvision.build_graph import build_graph
from circuitvision.constraints import validate
from circuitvision.motifs import all_motifs

out = build_graph(verbose=False)
comps, pins, nets = out['comps'], out['pins'], out['nets']
gray = out['gray']
rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'])
mo = all_motifs(comps, pins, nets, rep['ground_collapse']['merge_map'])
OUT = Path('demo/outputs')

# ---- 1. nets overlay (pin-bearing nets colored, rescued lime) ----
import cv2
labels = out['wire_ctx']['labels']
net_of, net_index = out['wire_ctx']['net_of'], out['wire_ctx']['net_index']
pinned_frags = {f for f, rt in net_of.items() if net_index[rt] in nets}
lab_color = np.zeros((*labels.shape, 3), np.uint8)
gnets = sorted(nets)
pal = np.random.default_rng(7).integers(40, 255, size=(len(gnets), 3))
gi = {n: i for i, n in enumerate(gnets)}
keep = np.isin(labels, list(pinned_frags))
for f in pinned_frags:
    lab_color[labels == f] = pal[gi[net_index[net_of[f]]] % len(pal)]
base = np.array(gray.convert('RGB'))
im = Image.fromarray(np.where(keep[..., None], lab_color, base).astype(np.uint8))
d = ImageDraw.Draw(im)
for dx, dy, _ in out['wire_ctx']['dots']:
    d.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], outline='white', width=2)
sizes = {}
for f in pinned_frags:
    ni = net_index[net_of[f]]
    sizes[ni] = sizes.get(ni, 0) + int((labels == f).sum())
for ni in sorted(sizes, key=lambda k: -sizes[k])[:12]:
    f0 = next(f for f in pinned_frags if net_index[net_of[f]] == ni)
    ys, xs = np.nonzero(labels == f0)
    d.text((int(xs.mean()), int(ys.mean())), f'N{ni}', fill='white',
           stroke_width=2, stroke_fill='black')
im.save(OUT / 'graph_nets.jpg')

# ---- 2. motif overlay ----
im2 = gray.convert('RGB')
d2 = ImageDraw.Draw(im2)
colors = {'voltage_divider': 'orange', 'decoupling': 'magenta',
          'rc_lowpass': 'cyan', 'common_emitter': 'lime',
          'parallel_pair': 'yellow'}
for name, color in colors.items():
    for v in mo[name]:
        members = [v[k] for k in ('r_top', 'r_bot', 'cap', 'series',
                                  'shunt', 'q', 'collector_r') if k in v]
        if 'diodes' in v:
            members = v['diodes']
        xs0 = min(comps[c][1][0] for c in members)
        ys0 = min(comps[c][1][1] for c in members)
        xs1 = max(comps[c][1][2] for c in members)
        ys1 = max(comps[c][1][3] for c in members)
        d2.rectangle([xs0 - 8, ys0 - 8, xs1 + 8, ys1 + 8], outline=color,
                     width=3)
        d2.text((xs0 - 8, ys0 - 20), name.split('_')[0], fill=color,
                stroke_width=2, stroke_fill='black')
im2.save(OUT / 'graph_motifs.jpg')

# ---- 3. subgraph diagram around net 386 ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FOCUS = 386
fcomps_on = sorted({pins[p][0] for p in nets.get(FOCUS, [])})
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title(f'subgraph: net 386 ({len(nets.get(FOCUS, []))} pins, GND-merged REF network)')
ys = np.linspace(0, 1, max(len(fcomps_on), 1))
pos = {}
for i, ci in enumerate(fcomps_on):
    pos[f'c{ci}'] = (0.15, ys[i] if len(fcomps_on) > 1 else 0.5)
    ax.text(0.15, ys[i] if len(fcomps_on) > 1 else 0.5,
            f'{comps[ci][0]}#{ci}', ha='right', va='center', fontsize=9,
            bbox=dict(fc='lightblue', ec='none', pad=2))
ax.text(0.85, 0.5, 'NET 386', ha='left', va='center', fontsize=11,
        bbox=dict(fc='orange', ec='none', pad=4))
mypins = [(p, pins[p][1]) for p in nets.get(FOCUS, [])]
ysp = np.linspace(0, 1, max(len(mypins), 1))
for i, (p, role) in enumerate(mypins):
    y = ysp[i] if len(mypins) > 1 else 0.5
    ci = pins[p][0]
    ax.plot([0.3, 0.7], [pos[f'c{ci}'][1], y], 'k-', lw=0.8)
    ax.plot([0.3, 0.7], [pos[f'c{ci}'][1], y], 'ko', ms=3)
    ax.text(0.5, y, role, ha='center', va='bottom', fontsize=8, color='red')
ax.plot([0.7, 0.7], [0, 1], color='orange', lw=2)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.05)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'graph_sub386.png', dpi=100)
print('saved graph_nets.jpg graph_motifs.jpg graph_sub386.png')
print(f'nets: {len(nets)} pin-bearing; top sizes:',
      sorted(((len(v), k) for k, v in nets.items()), reverse=True)[:8])
