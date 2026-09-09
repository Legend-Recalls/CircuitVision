"""Count job: per-source image counts per split (filenames only, no reads).

Source is encoded in the filename prefix, e.g. cghd.yolov8_train_....jpg
Output: source_counts.csv in version outputs + printed table.
"""
import csv
import os
from collections import Counter
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SOURCES = ["cghd.yolov8", "ci2n.device_identification.yolov8",
           "digitize_hcd.component_symbols.yolov8", "schematic.merged.yolov8",
           "schematic_images.hf.yolov8", "v3qwe.v2i.yolov8"]


def find_roots():
    roots = []
    for cur, dirs, _ in os.walk(INPUT):
        if os.path.basename(cur) in ("images", "labels", "orientation_labels"):
            dirs[:] = []
            continue
        if "train" in dirs and (Path(cur) / "train" / "images").is_dir():
            roots.append(Path(cur))
            dirs[:] = [d for d in dirs if d not in ("train", "valid", "test")]
    return roots


def main():
    roots = find_roots()
    assert roots, "no YOLO roots"
    totals = Counter()
    with open(WORK / "source_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root", "source", "split", "images"])
        for root in roots:
            for split in ("train", "valid", "test"):
                idir = root / split / "images"
                if not idir.is_dir():
                    continue
                c = Counter()
                for img in os.scandir(idir):
                    src = next((s for s in SOURCES if img.name.startswith(s)), "other")
                    c[src] += 1
                for src, n in sorted(c.items()):
                    w.writerow([root.name, src, split, n])
                    totals[(src, split)] += n
                print(f"{root.name}/{split}: {dict(sorted(c.items()))}", flush=True)
    print("=== TOTALS (source x split) ===", flush=True)
    grand = Counter()
    for (src, split), n in sorted(totals.items()):
        print(f"{src:40s} {split:6s} {n:8d}", flush=True)
        grand[src] += n
    print("=== GRAND ===", flush=True)
    for src, n in sorted(grand.items(), key=lambda x: -x[1]):
        print(f"{src:40s} {n:8d}", flush=True)


if __name__ == "__main__":
    main()
