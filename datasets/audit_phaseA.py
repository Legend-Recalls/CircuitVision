#!/usr/bin/env python3
"""Phase A dataset audit for ALL_COMPONENTS.merged.yolov8.

Checks (no external deps):
  1. Per-class instance histogram (train/val/test + total)
  2. Per-source image counts per split
  3. CGHD circuit-level leak across splits (from filenames: drafter_X_cYY_dZ_pW)
  4. Annotation sanity: zero-area / out-of-bounds / malformed label lines

Writes: audit_report.txt and class_histogram.csv next to this script.
"""
import os
import re
import sys
from collections import defaultdict, Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALL_COMPONENTS.merged.yolov8")
SPLITS = ["train", "valid", "test"]


def load_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
    if not m:
        sys.exit("could not parse names from data.yaml")
    return re.findall(r"'([^']*)'", m.group(1))


NAMES = load_names(os.path.join(ROOT, "data.yaml"))
NC = len(NAMES)

SRC_RE = re.compile(r"^(.*?)_(train|valid|test)_")
CGHD_RE = re.compile(r"drafter_(\d+)_c(\d+)_d(\d+)_p(\d+)")

class_counts = {s: Counter() for s in SPLITS}
src_img_counts = {s: Counter() for s in SPLITS}
cghd_circuit_splits = defaultdict(set)
cghd_circuit_files = defaultdict(list)
sanity = Counter()
sanity_examples = defaultdict(list)


def source_of(fname):
    m = SRC_RE.match(fname)
    return m.group(1) if m else "UNKNOWN"


for split in SPLITS:
    img_dir = os.path.join(ROOT, split, "images")
    lbl_dir = os.path.join(ROOT, split, "labels")
    if not os.path.isdir(img_dir):
        continue
    for fname in os.listdir(img_dir):
        src = source_of(fname)
        src_img_counts[split][src] += 1
        cg = CGHD_RE.search(fname)
        if cg and src.startswith("cghd"):
            key = f"drafter{cg.group(1)}_c{cg.group(2)}"
            cghd_circuit_splits[key].add(split)
            if len(cghd_circuit_files[key]) < 12:
                cghd_circuit_files[key].append((split, fname))
        stem = os.path.splitext(fname)[0]
        lpath = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lpath):
            sanity["missing_label_file"] += 1
            continue
        with open(lpath, "r", encoding="utf-8", errors="replace") as lf:
            for ln, line in enumerate(lf):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    sanity["malformed_line"] += 1
                    if len(sanity_examples["malformed_line"]) < 10:
                        sanity_examples["malformed_line"].append(f"{fname}:{ln} -> {line!r}")
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    sanity["parse_error"] += 1
                    continue
                if cid < 0 or cid >= NC:
                    sanity["class_id_out_of_range"] += 1
                    continue
                class_counts[split][cid] += 1
                if w <= 0 or h <= 0:
                    sanity["zero_area_box"] += 1
                    if len(sanity_examples["zero_area_box"]) < 10:
                        sanity_examples["zero_area_box"].append(f"{fname}:{ln}")
                if (x - w / 2) < -0.01 or (y - h / 2) < -0.01 or (x + w / 2) > 1.01 or (y + h / 2) > 1.01:
                    sanity["out_of_bounds_box"] += 1
                    if len(sanity_examples["out_of_bounds_box"]) < 10:
                        sanity_examples["out_of_bounds_box"].append(f"{fname}:{ln}")

totals = Counter()
for s in SPLITS:
    totals.update(class_counts[s])

hist_path = os.path.join(os.path.dirname(ROOT), "class_histogram.csv")
with open(hist_path, "w", encoding="utf-8") as f:
    f.write("class_id,class_name,train,valid,test,total\n")
    for cid in range(NC):
        f.write(f"{cid},{NAMES[cid]},{class_counts['train'][cid]},"
                f"{class_counts['valid'][cid]},{class_counts['test'][cid]},{totals[cid]}\n")

rep_path = os.path.join(os.path.dirname(ROOT), "audit_report.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    def out(*a):
        print(*a)
        f.write(" ".join(str(x) for x in a) + "\n")

    out("=" * 70)
    out("PHASE A DATASET AUDIT:", ROOT)
    out("=" * 70)
    out(f"classes: {NC}, total instances: {sum(totals.values()):,}")
    out("")
    out("--- 2. PER-SOURCE IMAGE COUNTS PER SPLIT ---")
    all_srcs = sorted(set().union(*[set(src_img_counts[s]) for s in SPLITS]))
    out(f"{'source':<40}{'train':>9}{'valid':>9}{'test':>9}{'total':>9}")
    for src in all_srcs:
        tr, va, te = src_img_counts['train'][src], src_img_counts['valid'][src], src_img_counts['test'][src]
        out(f"{src:<40}{tr:>9}{va:>9}{te:>9}{tr + va + te:>9}")
    out("")
    out("--- 1. CLASS HISTOGRAM (sorted ascending by total) ---")
    out(f"{'id':>4} {'name':<38}{'train':>9}{'valid':>9}{'test':>9}{'total':>9}")
    under100, under30, zero = [], [], []
    for cid in sorted(range(NC), key=lambda c: totals[c]):
        t = totals[cid]
        out(f"{cid:>4} {NAMES[cid]:<38}{class_counts['train'][cid]:>9}"
            f"{class_counts['valid'][cid]:>9}{class_counts['test'][cid]:>9}{t:>9}")
        if t == 0:
            zero.append(NAMES[cid])
        elif t < 30:
            under30.append(NAMES[cid])
        elif t < 100:
            under100.append(NAMES[cid])
    out("")
    out(f"ZERO-instance classes ({len(zero)}): {zero}")
    out(f"UNDER-30 (drop/fold candidates) ({len(under30)}): {under30}")
    out(f"UNDER-100 (augment-needed) ({len(under100)}): {under100}")
    out("")
    out("--- 3. CGHD CIRCUIT-LEVEL LEAK CHECK ---")
    leaked = {k: v for k, v in cghd_circuit_splits.items() if len(v) > 1}
    out(f"CGHD circuits seen: {len(cghd_circuit_splits)}")
    out(f"CGHD circuits spanning >1 split (LEAK): {len(leaked)}")
    for k in sorted(leaked)[:25]:
        out(f"  {k}: splits={sorted(leaked[k])}  e.g. {cghd_circuit_files[k][:2]}")
    if len(leaked) > 25:
        out(f"  ... and {len(leaked) - 25} more")
    out("")
    out("--- 4. ANNOTATION SANITY ---")
    if not sanity:
        out("no issues found")
    for k, v in sanity.most_common():
        out(f"  {k}: {v}")
        for ex in sanity_examples.get(k, [])[:5]:
            out(f"      e.g. {ex}")

print(f"\nWrote: {rep_path}")
print(f"Wrote: {hist_path}")
