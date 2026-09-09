"""Histogram job: per-class instance + image counts for the CLAHE YOLO dataset.

Runs on CPU in minutes. Output: histogram.csv in version outputs + printed table.
Attach dataset: muzaafnameerfirdausi/circuitvision-clahe
"""
import csv
import os
from collections import Counter
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find_roots():
    roots = []
    for cur, dirs, _ in os.walk(INPUT):
        base = os.path.basename(cur)
        if base in ("images", "labels", "orientation_labels"):
            dirs[:] = []
            continue
        if "train" in dirs and (Path(cur) / "train" / "images").is_dir():
            roots.append(Path(cur))
            dirs[:] = [d for d in dirs if d not in ("train", "valid", "test")]
    return roots


def names_for(root):
    for y in [root / "data.yaml", *sorted(root.rglob("data.yaml"))]:
        if y.is_file():
            names = []
            in_names = False
            for line in y.read_text().splitlines():
                s = line.strip()
                if s.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if s.startswith("- "):
                        names.append(s[2:].strip())
                    elif s and not s.startswith("#"):
                        break
            if names:
                return names
    return []


def main():
    roots = find_roots()
    assert roots, "no YOLO roots under /kaggle/input"
    print("roots:", [str(r) for r in roots])
    out_rows = []
    for root in roots:
        inst = Counter()
        imgs = Counter()
        n_img = 0
        for split in ("train", "valid", "test"):
            ldir = root / split / "labels"
            if not ldir.is_dir():
                continue
            for lf in ldir.glob("*.txt"):
                n_img += 1
                seen = set()
                for line in lf.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cid = int(line.split()[0])
                    except ValueError:
                        continue
                    inst[cid] += 1
                    seen.add(cid)
                for cid in seen:
                    imgs[cid] += 1
        names = names_for(root)
        print(f"--- {root} images={n_img} instances={sum(inst.values())} classes={len(names)}")
        rows = [(cid, names[cid] if cid < len(names) else f"id_{cid}",
                 inst.get(cid, 0), imgs.get(cid, 0)) for cid in range(max(len(names), max(inst, default=-1) + 1))]
        rows.sort(key=lambda r: r[2])
        for cid, nm, ni, nim in rows:
            print(f"{cid:4d} {nm:45s} inst={ni:8d} imgs={nim:7d}")
        out = WORK / f"histogram_{root.name}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["root", "class_id", "name", "instances", "images"])
            for cid, nm, ni, nim in rows:
                w.writerow([root.name, cid, nm, ni, nim])
        print("wrote", out)
        out_rows.append(out)
    print("done:", [str(o) for o in out_rows])


if __name__ == "__main__":
    main()
