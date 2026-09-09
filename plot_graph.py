"""Plot the graph itself: component <-> net bipartite node-link diagrams."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import networkx as nx
from build_graph import build_graph
from constraints import validate

out = build_graph(verbose=False)
comps, pins, nets = out['comps'], out['pins'], out['nets']
rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'])
gm = rep['ground_collapse']['merge_map']
en = lambda n: 'GND' if n in gm else f'N{n}'

G = nx.Graph()
cls_of = {}
for ci, (cls, _) in enumerate(comps):
    G.add_node(f'C{ci}', kind='comp', cls=cls)
    cls_of[f'C{ci}'] = cls
for n, ps in nets.items():
    G.add_node(en(n), kind='net', size=len(ps))
for pi, (ci, role, n) in enumerate(pins):
    if n is not None:
        G.add_edge(f'C{ci}', en(n), role=role)

print(f'nodes={G.number_of_nodes()} edges={G.number_of_edges()} '
      f'components={len(comps)} nets={len(nets)}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

classes = sorted({c for _, (c, _) in enumerate(comps)} |
                 {cls for cls, _ in comps})
cmap = plt.cm.tab20
col = {c: cmap(i % 20) for i, c in enumerate(sorted({cl for cl, _ in comps}))}


def draw(H, path, title, seed=7, fs=(20, 14), dots=18):
    pos = nx.spring_layout(H, seed=seed, k=0.6, iterations=80)
    fig, ax = plt.subplots(figsize=fs)
    cn = [v for v, d in H.nodes(data=True) if d['kind'] == 'comp']
    nn = [v for v, d in H.nodes(data=True) if d['kind'] != 'comp']
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=cn,
                           node_color=[col[H.nodes[v]['cls']] for v in cn],
                           node_size=dots, linewidths=0)
    nsizes = [40 + 30 * H.nodes[v]['size'] for v in nn]
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=nn, node_shape='s',
                           node_color='black', node_size=nsizes,
                           linewidths=0)
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.4, alpha=0.5)
    big = [v for v in nn if H.nodes[v]['size'] >= 6]
    nx.draw_networkx_labels(H, pos, ax=ax, labels={v: v for v in big},
                            font_size=7, font_color='red')
    ax.legend(handles=[Patch(color=col[c], label=c) for c in sorted(col)],
              fontsize=6, ncol=4, loc='upper left')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print('saved', path)


OUT = Path('demo/outputs')
draw(G, OUT / 'graph_plot.png',
     f'full graph: {len(comps)} comps - {G.number_of_edges()} links - {len(nets)} nets (squares, size=pins; GND merged)')

# backbone: nets with >=3 pins + attached comps
keep_nets = {n for n, d in G.nodes(data=True)
             if d['kind'] != 'comp' and d['size'] >= 3}
keep = set(keep_nets)
for n in keep_nets:
    keep |= set(G.neighbors(n))
H = G.subgraph(keep).copy()
draw(H, OUT / 'graph_backbone.png',
     f'backbone: nets>=3 pins ({len(keep_nets)} nets, {H.number_of_nodes()} nodes)',
     seed=11, fs=(18, 12), dots=60)
