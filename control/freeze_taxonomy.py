"""Freeze the shortened taxonomy from the CLAHE-92 histogram.

Policy (decision, frozen 2026-09-08):
- The CLAHE set is already the 106->92 remap. Shorten further by dropping
  classes the detector cannot learn: <100 instances (20 classes) plus any
  class in <25 images (id_59/probe, 16 images).
- Result: 71 classes, contiguous new ids. Everything else kept: for a
  detection-first pipeline, vocabulary recall beats a smaller label space,
  and the Stage-5+ correction engine needs the symbols present.
- NOT dropped for rarity alone: mid-tail 100-500 instance classes stay.

Outputs (runs/taxonomy_v1/):
- histogram_named.csv  (id, name, instances, images, decision)
- freeze.csv           (old_id, name, instances, keep[0/1], new_id[-1 if dropped])
- notes.md             (this rationale, n classes)
"""
import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
ap = argparse.ArgumentParser()
ap.add_argument("--hist", default="runs/hist_v1/histogram_circuitvision-clahe.csv")
ap.add_argument("--yaml", default="runs/hist_v1/data.clahe92.yaml")
ap.add_argument("--out", default="runs/taxonomy_v1")
ap.add_argument("--tag", default="v1")
ap.add_argument("--min-inst", type=int, default=100)
ap.add_argument("--min-imgs", type=int, default=25)
a = ap.parse_args()
HIST = ROOT / a.hist
YAML = ROOT / a.yaml
OUT = ROOT / a.out
OUT.mkdir(parents=True, exist_ok=True)

text = YAML.read_text()
m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
assert len(names) == 92, len(names)

rows = []
with open(HIST) as f:
    for r in csv.DictReader(f):
        cid = int(r["class_id"])
        rows.append({"id": cid, "name": names[cid],
                     "instances": int(r["instances"]), "images": int(r["images"])})
assert len(rows) == 92

for r in rows:
    r["keep"] = not (r["instances"] < a.min_inst or r["images"] < a.min_imgs)

kept = sorted([r for r in rows if r["keep"]], key=lambda r: r["id"])
for new_id, r in enumerate(kept):
    r["new_id"] = new_id
for r in rows:
    if not r["keep"]:
        r["new_id"] = -1

with open(OUT / "histogram_named.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class_id", "name", "instances", "images", "decision"])
    for r in sorted(rows, key=lambda r: r["instances"]):
        w.writerow([r["id"], r["name"], r["instances"], r["images"],
                    "keep" if r["keep"] else "drop"])

with open(OUT / "freeze.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["old_id", "name", "instances", "keep", "new_id"])
    for r in sorted(rows, key=lambda r: r["id"]):
        w.writerow([r["id"], r["name"], r["instances"], int(r["keep"]), r["new_id"]])

dropped = [r for r in rows if not r["keep"]]
(OUT / "notes.md").write_text(
    f"# Taxonomy freeze {a.tag} (2026-09-08)\n\n"
    f"- hist: {a.hist}\n"
    f"- kept: {len(kept)} / dropped: {len(dropped)}\n"
    f"- rule: drop instances<{a.min_inst} OR images<{a.min_imgs}\n"
    "- kept mid-tail (100-500 inst): recall matters more than precision for the pipeline\n"
    "- dominant classes (gnd/junction/text/and) stay; handle via loss weighting later, not by dropping\n\n"
    "## Dropped\n\n" +
    "".join(f"- {r['id']:3d} {r['name']:40s} inst={r['instances']:6d} imgs={r['images']:6d}\n" for r in sorted(dropped, key=lambda r: r["instances"])) +
    "\n## Use\n\n- `freeze.csv` old_id->new_id (-1 = dropped) is baked into jobs/train_stage1 remap step.\n"
)
print(f"kept={len(kept)} dropped={len(dropped)} -> {OUT}")
for r in sorted(dropped, key=lambda r: r["instances"]):
    print(f"  drop {r['id']:3d} {r['name']:40s} inst={r['instances']:6d} imgs={r['images']:6d}")
