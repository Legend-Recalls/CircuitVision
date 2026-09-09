"""CircuitVision: ANY schematic image -> graph -> netlist -> LTspice.

Usage: python pipeline.py <image> [--sim] [--ltspice] [--outdir DIR]
  --sim      also emit a sim deck (generic 12V rail on largest net,
             2mV collector kicks, .tran uic). Toy stimulus, stated.
  --ltspice  batch-run the deck in LTspice after building it.
Outputs: <outdir>/<stem>_dets.jpg, <stem>_nets.jpg, <stem>_graph.png,
         <stem>.sp (+ <stem>_drive.cir, .log, .raw with --sim/--ltspice).
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

from build_graph import build_graph
from constraints import validate
from motifs import all_motifs, rank_repairs, electrical_nets
from correct import apply_corrections
from netlist import build_netlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--ltspice', action='store_true')
    ap.add_argument('--outdir', default='demo/outputs')
    a = ap.parse_args()
    src = Path(a.image)
    assert src.exists(), f'missing {src}'
    tag = src.stem
    outdir = Path(a.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'== {tag}: detect -> pins -> nets ==')
    out = build_graph(src=str(src))
    comps, pins, nets = out['comps'], out['pins'], out['nets']
    gray = out['gray']
    print(Counter(c for c, _ in comps))

    im = gray.convert('RGB')
    d = ImageDraw.Draw(im)
    for cls, (x0, y0, x1, y1) in comps:
        d.rectangle([x0, y0, x1, y1], outline='lime', width=2)
        d.text((x0, max(0, y0 - 12)), cls, fill='lime')
    im.save(outdir / f'{tag}_dets.jpg')

    imp = gray.convert('RGB')
    dp = ImageDraw.Draw(imp)
    for (x, y) in out['pin_xy']:
        dp.ellipse([x - 5, y - 5, x + 5, y + 5], outline='red', width=2)
    imp.save(outdir / f'{tag}_pins.jpg')
    print(f'{len(out["pin_xy"])} pins -> {tag}_pins.jpg')

    from wire_nets import mask_interiors as _mask, binarize as _bin
    masked_all = _mask(_bin(gray), [b for _, b in comps])
    imw = Image.fromarray(((1 - (masked_all > 0)) * 255).astype(np.uint8)
                          ).convert('RGB')
    dw = ImageDraw.Draw(imw)
    for dx, dy, _ in out['wire_ctx']['dots']:
        dw.ellipse([dx - 4, dy - 4, dx + 4, dy + 4], outline='red', width=1)
    imw.save(outdir / f'{tag}_wires.jpg')
    print(f'wires ({int(masked_all.sum())} px) -> {tag}_wires.jpg')

    import cv2
    labels = out['wire_ctx']['labels']
    net_of, net_index = out['wire_ctx']['net_of'], out['wire_ctx']['net_index']
    pinned = {f for f, rt in net_of.items() if net_index[rt] in nets}
    lab_color = np.zeros((*labels.shape, 3), np.uint8)
    gnets = sorted(nets)
    pal = np.random.default_rng(7).integers(40, 255, size=(len(gnets), 3))
    gi = {n: i for i, n in enumerate(gnets)}
    keep = np.isin(labels, list(pinned))
    for f in pinned:
        lab_color[labels == f] = pal[gi[net_index[net_of[f]]] % len(pal)]
    base_img = np.array(gray.convert('RGB'))
    Image.fromarray(np.where(keep[..., None], lab_color, base_img)
                    .astype(np.uint8)).save(outdir / f'{tag}_nets.jpg')
    print(f'{len(comps)} comps, {len(pins)} pins, {len(nets)} pin-nets')

    print('== validate ==')
    rep = validate(comps, pins, nets, out['wire_ctx'], out['footprints'],
                   out['pin_xy'])
    for name, v in rep.items():
        if name == 'structural_candidates':
            print(f'[{name}]: {v}')
            continue
        fl = v.get('flags', '')
        print(f'[{name}]: {v["verdict"]} '
              f'flags={len(fl) if isinstance(fl, list) else 0}')

    print('== motifs ==')
    gm = rep['ground_collapse']['merge_map']
    mo = all_motifs(comps, pins, nets, gm)
    print({k: len(v) for k, v in mo.items()})

    print('== correct ==')
    enets = electrical_nets(pins, gm)
    isl = rep['shorts_islands']
    troubled = [(p, c, pins[p][1]) for p, c in isl['dangling_pins']]
    troubled += [(p, pins[p][0], pins[p][1]) for _, p in isl['island_nets']]
    rank = rank_repairs(comps, pins, out['pin_xy'], enets, troubled,
                        wire_ctx=out['wire_ctx'], gnd_merge=gm)
    fix = apply_corrections(comps, pins, out['pin_xy'], nets,
                            out['wire_ctx'], gm, rank, out['footprints'])
    n_fixed = sum(1 for x in fix['log'] if x['action'] == 'reassign')
    print(f'reassigns={n_fixed} removed={fix["removed"][0]} '
          f'{len(comps)}->{len(fix["comps"])} comps '
          f'{len(nets)}->{len(fix["nets"])} nets')
    c2, p2, n2 = fix['comps'], fix['pins'], fix['nets']

    print('== graph plot ==')
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    G = nx.Graph()
    for ci, (cls, _) in enumerate(c2):
        G.add_node(f'C{ci}', kind='comp', cls=cls)
    for n, ps in n2.items():
        G.add_node('GND' if n in gm else f'N{n}', kind='net', size=len(ps))
    for _, (ci, _, n) in enumerate(p2):
        if n is not None:
            G.add_edge(f'C{ci}', 'GND' if n in gm else f'N{n}')
    pos = nx.spring_layout(G, seed=7, k=0.6, iterations=80)
    fig, ax = plt.subplots(figsize=(20, 14))
    cn = [v for v, d in G.nodes(data=True) if d['kind'] == 'comp']
    nn = [v for v, d in G.nodes(data=True) if d['kind'] != 'comp']
    cmap = plt.cm.tab20
    clss = sorted({cl for cl, _ in c2})
    col = {c: cmap(i % 20) for i, c in enumerate(clss)}
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cn,
                           node_color=[col[G.nodes[v]['cls']] for v in cn],
                           node_size=18, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nn, node_shape='s',
                           node_color='black',
                           node_size=[40 + 30 * G.nodes[v]['size'] for v in nn],
                           linewidths=0)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.4, alpha=0.5)
    ax.set_title(f'{tag}: {len(c2)} comps, {G.number_of_edges()} links, '
                 f'{len(n2)} nets')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outdir / f'{tag}_graph.png', dpi=90)
    plt.close(fig)

    print('== netlist ==')
    rep2 = validate(c2, p2, n2, out['wire_ctx'], [],
                    out['pin_xy'])
    txt, vr = build_netlist(c2, p2, n2,
                            rep2['ground_collapse']['merge_map'],
                            title=tag)
    open(outdir / f'{tag}.sp', 'w', newline='\n').write(txt)
    print(f'{vr["modeled"]} elements, {vr["nodes"]} nodes, '
          f'bad={len(vr["bad_lines"])}, flt={vr["flt_pins"]}')

    print('== asc reconstruction ==')
    from asc_gen import gen_asc, verify_asc, placements
    exp = gen_asc(c2, p2, out['pin_xy'], n2,
                  rep2['ground_collapse']['merge_map'],
                  str(outdir / f'{tag}.asc'), f'{tag} via CircuitVision',
                  trees=True)
    vrf = verify_asc(str(outdir / f'{tag}.asc'), exp)
    if vrf['mismatch']:
        # snapped rails shorted something: fall back to unsnapped geometry
        exp = gen_asc(c2, p2, out['pin_xy'], n2,
                      rep2['ground_collapse']['merge_map'],
                      str(outdir / f'{tag}.asc'),
                      f'{tag} via CircuitVision', trees=True, snap_tol=0)
        vrf = verify_asc(str(outdir / f'{tag}.asc'), exp)
        print('snap reverted (mismatch), unsnapped: ', end='')
    print(f'asc: {len(exp)} pin-nets, netlister checked={vrf["checked"]} '
          f'mismatch={len(vrf["mismatch"])}')
    for m in vrf['mismatch'][:15]:
        print('   ASC-MISMATCH', m)
    plc, _notes = placements(c2, p2, out['pin_xy'])
    prv = gray.convert('RGB').resize((gray.width * 2, gray.height * 2))
    dpv = ImageDraw.Draw(prv)
    for inst, sym, X, Y, rot, sp, val in plc:
        if sym == 'RECT':
            continue
        xs = [s[1] for s in sp] + [X]
        ys = [s[2] for s in sp] + [Y]
        dpv.rectangle([2 * min(xs) - 6, 2 * min(ys) - 6, 2 * max(xs) + 6,
                       2 * max(ys) + 6], outline='blue', width=2)
        dpv.text((2 * X, 2 * Y - 24), f'{inst}({sym})', fill='blue')
    prv.save(outdir / f'{tag}_asc_preview.png')
    print(f'saved {tag}.asc + {tag}_asc_preview.png')
    from asc_render import render_asc
    render_asc(str(outdir / f'{tag}.asc'),
               str(outdir / f'{tag}_ltspice.png'))

    if a.sim or a.ltspice:
        en2 = lambda n: 'GND' if n in rep2['ground_collapse']['merge_map'] else n
        # supply rail: resistor/collector-loaded, never base- or cap-loaded
        # (largest-net picked the T1-base middle node on wien: wrong).
        score = {}
        for pi, (ci, role, n) in enumerate(p2):
            if n is None or en2(n) == 'GND':
                continue
            cls = c2[ci][0]
            w = (1 if cls == 'resistor' or role in ('c', 'd') else 0) - \
                (2 if role in ('b', 'g') or cls == 'capacitor' else 0)
            score[en2(n)] = score.get(en2(n), 0) + w
        vcc = max(score, key=score.get) if score else 'N0'
        kick = sorted({en2(n) for ci, r, n in p2
                       if c2[ci][0] in ('npn', 'pnp', 'transistor.bjt')
                       and r == 'c' and n is not None and en2(n) != 'GND'})
        deck = txt.splitlines()
        ins = next(i for i, l in enumerate(deck) if l.startswith('* --- models'))
        nm = lambda n: '0' if n == 'GND' else f'N{n}'
        deck = (deck[:ins] +
                [f'VVCC {nm(vcc)} 0 DC 12  ; TOY supply (placeholder)'] +
                [f'.ic V({nm(k)})=2m' for k in kick] +
                ['.tran 0 8m 0 uic'] + deck[ins:])
        deck = [l for l in deck if not l.startswith('.op')]
        open(outdir / f'{tag}_drive.cir', 'w', newline='\n').write(
            '\n'.join(deck) + '\n')
        print(f'deck: Vcc={vcc} kick={kick} -> {tag}_drive.cir')
        if a.ltspice:
            import subprocess
            exe = (r'C:\Users\wwwna\AppData\Local\Programs\ADI\LTspice'
                   r'\LTspice.exe')
            subprocess.run([exe, '-ascii', '-b',
                            str(outdir / f'{tag}_drive.cir')],
                           cwd=str(Path.cwd()), timeout=240,
                           capture_output=True)
            print('LTspice batch done ->', f'{tag}_drive.log/.raw')


if __name__ == '__main__':
    main()
