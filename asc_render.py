"""Render an LTspice .asc with its own symbol art (from lib.zip).

Draws the exact geometry LTspice shows (same .asy line art, R0/R90),
so the PNG previews what the viewer displays. Generic: any .asc using
stock symbols + WIRE/FLAG/TEXT/RECTANGLE.
"""
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

LIB = (r'C:\Users\wwwna\AppData\Local\Programs\ADI\LTspice\lib.zip')
_Z = None


def _lib():
    global _Z
    if _Z is None:
        _Z = zipfile.ZipFile(LIB)
    return _Z


def load_art(sym):
    """[(x0,y0,x1,y1), ...] line art in symbol coords (R0)."""
    try:
        txt = _lib().read(f'lib/sym/{sym}.asy').decode()
    except KeyError:
        return []
    art = []
    for ln in txt.splitlines():
        p = ln.split()
        if p and p[0] == 'LINE' and p[1] == 'Normal' and len(p) == 6:
            art.append(tuple(int(v) for v in p[2:6]))
    return art


def xform(pt, rot):
    x, y = pt
    return (-y, x) if rot == 'R90' else (x, y)


def render_asc(path, out=None, scale=1.0, margin=60):
    cmds = [l.rstrip('\n') for l in open(path)]
    syms, wires, flags, texts, rects = [], [], [], [], []
    for ln in cmds:
        p = ln.split()
        if not p:
            continue
        if p[0] == 'SYMBOL' and len(p) >= 5:
            syms.append((p[1], int(p[2]), int(p[3]), p[4]))
        elif p[0] == 'WIRE' and len(p) == 5:
            wires.append(tuple(int(v) for v in p[1:5]))
        elif p[0] == 'FLAG' and len(p) >= 4:
            flags.append((int(p[1]), int(p[2]), p[3]))
        elif p[0] == 'TEXT' and len(p) >= 6:
            x, y = int(p[1]), int(p[2])
            t = ln.split(None, 5)[5]
            texts.append((x, y, t))
        elif p[0] == 'RECTANGLE' and len(p) == 6:
            rects.append(tuple(int(v) for v in p[2:6]))
    xs = [c for s in (wires, [(f[0], f[1], f[0], f[1]) for f in flags])
          for c in s for c in (c[0], c[2])] or [0]
    ys = [c for s in (wires, [(f[0], f[1], f[0], f[1]) for f in flags])
          for c in s for c in (c[1], c[3])] or [0]
    for _, X, Y, _ in syms:
        xs += [X - 100, X + 100]
        ys += [Y - 100, Y + 100]
    x0, y0 = min(xs) - margin, min(ys) - margin
    ww, hh = (max(xs) - x0 + margin) * scale, (max(ys) - y0 + margin) * scale
    im = Image.new('RGB', (int(ww), int(hh)), 'white')
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()

    def P(x, y):
        return ((x - x0) * scale, (y - y0) * scale)

    for (x1, y1, x2, y2) in wires:
        d.line([P(x1, y1), P(x2, y2)], fill='black', width=2)
    art_cache = {}
    for sym, X, Y, rot in syms:
        if sym not in art_cache:
            art_cache[sym] = load_art(sym)
        for ax0, ay0, ax1, ay1 in art_cache[sym]:
            (bx0, by0), (bx1, by1) = xform((ax0, ay0), rot), \
                xform((ax1, ay1), rot)
            d.line([P(X + bx0, Y + by0), P(X + bx1, Y + by1)],
                   fill='black', width=2)
    for x, y, w, h in rects:
        d.rectangle([P(x, y), P(x + w, y + h)], outline='black', width=2)
    for x, y, name in flags:
        if name != '0':
            d.text((P(x, y)[0] + 5, P(x, y)[1] - 18), name, fill='darkblue',
                   font=font)
        else:
            px, py = P(x, y)
            d.line([(px, py), (px, py + 14)], fill='black', width=2)
            for i, wdt in enumerate((20, 13, 6)):
                yy = py + 14 + i * 6
                d.line([(px - wdt / 2, yy), (px + wdt / 2, yy)],
                       fill='black', width=2)
    for x, y, t in texts:
        if t.startswith(';'):
            continue
        if t.startswith('!'):
            continue
        d.text(P(x, y), t, fill='black', font=font)
    im.save(out or (str(path) + '.png'))
    print('rendered', out or (str(path) + '.png'), im.size)
    return im
