#!/usr/bin/env python3
"""Apply the Phase A taxonomy remap: 106 -> 92 classes.

Decisions (report/01_dataset_audit.md section 4, REVISED after orientation strip):
  - dflipflop/tgate are NOT dropped: the original audit undercounted them as 0
    because their instances lived in 6-column schematic_images labels.
    True counts: dflipflop 39,347 / tgate 39,366 -> KEEP.
  - FOLD .cross -> parent:      nmos.cross->nmos, pmos.cross->pmos,
                                npn.cross->npn,   pnp.cross->pnp
  - FOLD umbrella:              transistor -> transistor.bjt
  - FOLD under-30 -> unknown:   generator, connector, diode.thyrector,
                                galvanometer, operational_amplifier.schmitt_trigger,
                                frequency_meter, buzzer, ohmmeter, thermistor
Also drops zero-area boxes (w<=0 or h<=0), keeping orientation sidecars aligned.

Safety: backs up all label dirs to backup/pre_taxonomy_remap/<split>/ before
rewriting (folds are lossy, so the backup is the revert path).
Outputs: rewritten labels, new data.yaml (nc=92), TAXONOMY_REMAP.csv.
Default DRY-RUN. Pass --apply to modify files.
"""
import os
import re
import sys
import csv
import shutil
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALL_COMPONENTS.merged.yolov8")
SPLITS = ["train", "valid", "test"]
BACKUP = os.path.join(ROOT, "backup", "pre_taxonomy_remap")
APPLY = "--apply" in sys.argv

DROP = set()  # REVISED: dflipflop/tgate have ~39k instances each (audit undercount) - keep them
FOLD = {
    "nmos.cross": "nmos",
    "pmos.cross": "pmos",
    "npn.cross": "npn",
    "pnp.cross": "pnp",
    "transistor": "transistor.bjt",
    "generator": "unknown",
    "connector": "unknown",
    "diode.thyrector": "unknown",
    "galvanometer": "unknown",
    "operational_amplifier.schmitt_trigger": "unknown",
    "frequency_meter": "unknown",
    "buzzer": "unknown",
    "ohmmeter": "unknown",
    "thermistor": "unknown",
}


def load_names():
    with open(os.path.join(ROOT, "data.yaml"), "r", encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
    return re.findall(r"'([^']*)'", m.group(1)), txt


def main():
    old_names, yaml_txt = load_names()
    assert len(old_names) == 106, f"expected 106 classes, found {len(old_names)} (already remapped?)"

    removed = DROP | set(FOLD)
    new_names = [n for n in old_names if n not in removed]
    new_idx = {n: i for i, n in enumerate(new_names)}

    id_map = {}
    for oid, name in enumerate(old_names):
        if name in DROP:
            id_map[oid] = None
        elif name in FOLD:
            id_map[oid] = new_idx[FOLD[name]]
        else:
            id_map[oid] = new_idx[name]

    remapped = Counter()
    dropped_boxes = 0
    files_changed = 0
    new_counts = Counter()

    for split in SPLITS:
        lbl_dir = os.path.join(ROOT, split, "labels")
        side_dir = os.path.join(ROOT, "orientation_labels", split)
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            lpath = os.path.join(lbl_dir, fname)
            with open(lpath, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f.read().splitlines() if l.strip()]
            spath = os.path.join(side_dir, fname)
            sidecar = None
            if os.path.exists(spath):
                with open(spath, "r", encoding="utf-8") as sf:
                    sidecar = [l.strip() for l in sf.read().splitlines() if l.strip()]
            out_lines, out_side, changed = [], [], False
            for i, line in enumerate(lines):
                parts = line.split()
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                if w <= 0 or h <= 0:
                    dropped_boxes += 1
                    changed = True
                    continue
                nid = id_map.get(cid)
                if nid is None:
                    dropped_boxes += 1
                    changed = True
                    continue
                if old_names[cid] in FOLD:
                    remapped[old_names[cid]] += 1
                if nid != cid:
                    changed = True
                new_counts[new_names[nid]] += 1
                out_lines.append(" ".join([str(nid)] + parts[1:5]))
                if sidecar is not None and i < len(sidecar):
                    out_side.append(sidecar[i])
            if changed:
                files_changed += 1
                if APPLY:
                    bdir = os.path.join(BACKUP, split)
                    os.makedirs(bdir, exist_ok=True)
                    bpath = os.path.join(bdir, fname)
                    if not os.path.exists(bpath):
                        shutil.copy2(lpath, bpath)
                    with open(lpath, "w", encoding="utf-8") as f:
                        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
                    if sidecar is not None:
                        with open(spath, "w", encoding="utf-8") as sf:
                            sf.write("\n".join(out_side) + ("\n" if out_side else ""))

    mode = "APPLY" if APPLY else "DRY-RUN"
    rep = [f"=== Taxonomy remap 106 -> {len(new_names)} [{mode}] ===", ""]
    rep.append(f"classes removed: {len(removed)} (drop {len(DROP)}, fold {len(FOLD)})")
    rep.append(f"label files changed: {files_changed}")
    rep.append(f"boxes dropped (zero-area or dropped class): {dropped_boxes}")
    rep.append("")
    rep.append("instances redirected per folded class:")
    for name, cnt in sorted(remapped.items()):
        rep.append(f"  {name:<40} -> {FOLD[name]:<20} {cnt}")
    rep.append("")
    rep.append("sanity (new counts for fold targets):")
    for tgt in sorted(set(FOLD.values())):
        rep.append(f"  {tgt}: {new_counts[tgt]}")

    if APPLY:
        names_str = ", ".join(f"'{n}'" for n in new_names)
        new_yaml = re.sub(r"nc:\s*\d+", f"nc: {len(new_names)}", yaml_txt)
        new_yaml = re.sub(r"names:\s*\[.*?\]", f"names: [{names_str}]", new_yaml, flags=re.S)
        new_yaml = new_yaml.rstrip() + ("\nremap_notes: taxonomy remap 106->%d applied; see TAXONOMY_REMAP.csv\n" % len(new_names))
        with open(os.path.join(ROOT, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(new_yaml)
        with open(os.path.join(ROOT, "TAXONOMY_REMAP.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["old_id", "old_name", "new_id", "new_name", "action"])
            for oid, name in enumerate(old_names):
                nid = id_map[oid]
                if nid is None:
                    w.writerow([oid, name, "", "", "drop"])
                elif name in FOLD:
                    w.writerow([oid, name, nid, FOLD[name], "fold"])
                else:
                    w.writerow([oid, name, nid, name, "keep"])
        rep.append("")
        rep.append(f"new data.yaml written (nc={len(new_names)})")
        rep.append(f"mapping: {os.path.join(ROOT, 'TAXONOMY_REMAP.csv')}")
        rep.append(f"backups: {BACKUP}")
    else:
        rep.append("")
        rep.append("Dry-run only. Re-run with --apply to modify files.")

    text = "\n".join(rep)
    print(text)
    if APPLY:
        rdir = os.path.join(os.path.dirname(ROOT), "report")
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "04_taxonomy_remap.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
