"""Structure probe: what does the published focused69 dataset actually contain?

Lists tree layout + counts under /kaggle/input (zips? extracted tree?),
reads data.yaml nc/names. Read-only, CPU, minutes.
"""
import os
from pathlib import Path

INPUT = Path("/kaggle/input")


def main():
    for p in sorted(INPUT.iterdir()):
        print(f"top: {p} dir={p.is_dir()}", flush=True)
        if p.is_dir():
            for q in sorted(p.iterdir()):
                print(f"  sub: {q} dir={q.is_dir()}", flush=True)
                if q.is_dir():
                    kids = sorted(q.iterdir())
                    for r in kids[:15]:
                        print(f"    - {r} dir={r.is_dir()}", flush=True)
                    if len(kids) > 15:
                        print(f"    ... +{len(kids) - 15} more", flush=True)
    roots = [p for p in INPUT.rglob("*") if p.is_dir() and (p / "train" / "images").is_dir()]
    print("yolo roots:", [str(p) for p in roots], flush=True)
    ds = next((p for p in roots if "focused69" in str(p)), roots[0] if roots else None)
    if ds is None:
        print("NO yolo root attached; done", flush=True)
        return
    print("using:", ds, flush=True)
    n_zip = n_img = n_lab = 0
    for cur, dirs, files in os.walk(ds):
        for f in files:
            fp = Path(cur) / f
            if f.endswith(".zip"):
                n_zip += 1
                print(f"zip: {fp.relative_to(ds)} {fp.stat().st_size / 1e9:.2f} GB", flush=True)
            elif (Path(cur).name == "images"):
                n_img += 1
            elif (Path(cur).name == "labels"):
                n_lab += 1
    print(f"zips={n_zip} images={n_img} labels={n_lab}", flush=True)
    for y in sorted(ds.rglob("data.yaml")):
        print(f"--- {y.relative_to(ds)} ---", flush=True)
        txt = y.read_text().splitlines()
        print(f"lines={len(txt)} nc_line={[l for l in txt if l.startswith('nc:')]}", flush=True)
        print("first names:", [l for l in txt if l.startswith("- ")][:5], flush=True)


if __name__ == "__main__":
    main()
