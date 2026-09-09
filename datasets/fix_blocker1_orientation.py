#!/usr/bin/env python3
"""Fix Blocker 1: strip the 6th orientation token from schematic_images.hf labels.

YOLOv8 detection requires exactly 5 columns per label line:
    class cx cy w h
schematic_images.hf labels carry a 6th orientation token (R0/R90/.../MXR90).
This script:
  - finds every label file whose lines have 6 columns,
  - backs up the ORIGINAL file (once) under backup/orientation_strip/<split>/,
  - writes the orientation token to a sidecar: orientation_labels/<split>/<stem>.txt
    (one token per box line, same order as the label file),
  - rewrites the label file with 5 columns.

Default is DRY-RUN (reports only). Pass --apply to modify files.
Idempotent: files already at 5 columns are skipped.
Reversible: restore from backup/orientation_strip/.
"""
import os
import sys
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALL_COMPONENTS.merged.yolov8")
SPLITS = ["train", "valid", "test"]
BACKUP = os.path.join(ROOT, "backup", "orientation_strip")
SIDECAR = os.path.join(ROOT, "orientation_labels")
APPLY = "--apply" in sys.argv


def process():
    stats = {s: {"scanned": 0, "fixed": 0, "already5": 0, "bad": 0} for s in SPLITS}
    bad_examples = []
    for split in SPLITS:
        lbl_dir = os.path.join(ROOT, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            stats[split]["scanned"] += 1
            lpath = os.path.join(lbl_dir, fname)
            with open(lpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            has6 = False
            new_lines, orients, ok = [], [], True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    new_lines.append(line)
                    orients.append("R0")  # default for non-oriented sources
                elif len(parts) == 6:
                    has6 = True
                    new_lines.append(" ".join(parts[:5]))
                    orients.append(parts[5])
                else:
                    ok = False
                    break
            if not ok:
                stats[split]["bad"] += 1
                if len(bad_examples) < 10:
                    bad_examples.append(f"{split}/{fname}")
                continue
            if not has6:
                stats[split]["already5"] += 1
                continue
            stats[split]["fixed"] += 1
            if APPLY:
                bdir = os.path.join(BACKUP, split)
                os.makedirs(bdir, exist_ok=True)
                bpath = os.path.join(bdir, fname)
                if not os.path.exists(bpath):
                    shutil.copy2(lpath, bpath)
                sdir = os.path.join(SIDECAR, split)
                os.makedirs(sdir, exist_ok=True)
                with open(os.path.join(sdir, fname), "w", encoding="utf-8") as sf:
                    sf.write("\n".join(orients) + ("\n" if orients else ""))
                with open(lpath, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(new_lines) + ("\n" if new_lines else ""))
    return stats, bad_examples


def main():
    stats, bad = process()
    mode = "APPLY" if APPLY else "DRY-RUN"
    lines = [f"=== Blocker 1 orientation strip [{mode}] ===", ""]
    tot_fixed = 0
    for s in SPLITS:
        st = stats[s]
        tot_fixed += st["fixed"]
        lines.append(f"{s}: scanned={st['scanned']} to_fix={st['fixed']} "
                     f"already5={st['already5']} bad={st['bad']}")
    lines.append("")
    lines.append(f"TOTAL files needing fix: {tot_fixed}")
    if bad:
        lines.append(f"BAD (unexpected column count) examples: {bad}")
    if APPLY:
        lines.append(f"Backups: {BACKUP}")
        lines.append(f"Orientation sidecar: {SIDECAR}")
    else:
        lines.append("Dry-run only. Re-run with --apply to modify files.")
    report = "\n".join(lines)
    print(report)
    if APPLY:
        rdir = os.path.join(os.path.dirname(ROOT), "report")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "02_orientation_strip.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
