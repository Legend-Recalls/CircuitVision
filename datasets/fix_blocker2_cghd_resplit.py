#!/usr/bin/env python3
"""Fix Blocker 2: re-split CGHD by circuit so no circuit leaks across splits.

CGHD filenames encode the circuit: ..._drafter_<D>_c<C>_d<Dr>_p<P>.<ext>
All photos sharing drafter<D>_c<C> belong to ONE circuit and must go to a
single split. Assignment is deterministic (md5 of circuit key -> bucket),
target ratio ~70/15/15, so re-runs are stable.

Only CGHD files are touched. Each move relocates BOTH the image and its label
(and orientation sidecar if present) to the target split. A manifest
(report/03_cghd_resplit_manifest.csv) records every move for reversibility.

Default DRY-RUN. Pass --apply to move files.
"""
import os
import re
import sys
import csv
import hashlib
import shutil
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALL_COMPONENTS.merged.yolov8")
SPLITS = ["train", "valid", "test"]
CGHD_RE = re.compile(r"drafter_(\d+)_c_?(\d+)_d(\d+)_p(\d+)")
APPLY = "--apply" in sys.argv
REPORT_DIR = os.path.join(os.path.dirname(ROOT), "report")


def bucket(circuit_key):
    h = int(hashlib.md5(circuit_key.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "valid"
    return "test"


def img_for_label(split, stem):
    idir = os.path.join(ROOT, split, "images")
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(idir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    circuit_files = defaultdict(list)
    for split in SPLITS:
        idir = os.path.join(ROOT, split, "images")
        if not os.path.isdir(idir):
            continue
        for fname in os.listdir(idir):
            if not fname.lower().startswith("cghd"):
                continue
            m = CGHD_RE.search(fname)
            if not m:
                continue
            key = f"drafter{m.group(1)}_c{m.group(2)}"
            stem = os.path.splitext(fname)[0]
            circuit_files[key].append((split, stem))

    moves = []
    target_counts = {s: 0 for s in SPLITS}
    for key, files in circuit_files.items():
        tgt = bucket(key)
        for cur_split, stem in files:
            target_counts[tgt] += 1
            if cur_split != tgt:
                moves.append((stem, cur_split, tgt))

    mode = "APPLY" if APPLY else "DRY-RUN"
    report = [f"=== Blocker 2 CGHD re-split [{mode}] ===", ""]
    report.append(f"CGHD circuits: {len(circuit_files)}")
    report.append(f"CGHD photos total: {sum(len(v) for v in circuit_files.values())}")
    report.append(f"Target per-split CGHD counts (by circuit): "
                  f"train={target_counts['train']} valid={target_counts['valid']} test={target_counts['test']}")
    report.append(f"Files to move: {len(moves)}")
    report.append("")

    os.makedirs(REPORT_DIR, exist_ok=True)
    manifest = os.path.join(REPORT_DIR, "03_cghd_resplit_manifest.csv")

    if APPLY:
        moved, missing_img = 0, 0
        with open(manifest, "w", newline="", encoding="utf-8") as mf:
            w = csv.writer(mf)
            w.writerow(["stem", "from_split", "to_split", "moved_image", "moved_label", "moved_sidecar"])
            for stem, frm, to in moves:
                src_img = img_for_label(frm, stem)
                mi = ml = ms = 0
                if src_img:
                    dst_img = os.path.join(ROOT, to, "images", os.path.basename(src_img))
                    shutil.move(src_img, dst_img)
                    mi = 1
                else:
                    missing_img += 1
                src_lbl = os.path.join(ROOT, frm, "labels", stem + ".txt")
                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, os.path.join(ROOT, to, "labels", stem + ".txt"))
                    ml = 1
                src_side = os.path.join(ROOT, "orientation_labels", frm, stem + ".txt")
                if os.path.exists(src_side):
                    os.makedirs(os.path.join(ROOT, "orientation_labels", to), exist_ok=True)
                    shutil.move(src_side, os.path.join(ROOT, "orientation_labels", to, stem + ".txt"))
                    ms = 1
                w.writerow([stem, frm, to, mi, ml, ms])
                moved += 1
        report.append(f"Moved: {moved} (missing images: {missing_img})")
        report.append(f"Manifest: {manifest}")
    else:
        report.append("Dry-run only. Re-run with --apply to move files.")
        report.append("Sample planned moves:")
        for stem, frm, to in moves[:10]:
            report.append(f"  {frm} -> {to}: {stem}")

    text = "\n".join(report)
    print(text)
    if APPLY:
        with open(os.path.join(REPORT_DIR, "03_cghd_resplit.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
