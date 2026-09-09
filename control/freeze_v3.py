"""Freeze v3: v2 minus `unknown` (69 -> 68 classes)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "runs" / "taxonomy_v2" / "freeze.csv").open()))
assert len(rows) == 92, len(rows)
for r in rows:
    if r["name"] == "unknown":
        r["keep"] = "0"
kept = sorted([r for r in rows if int(r["keep"])], key=lambda r: int(r["old_id"]))
for i, r in enumerate(kept):
    r["new_id"] = i
for r in rows:
    if not int(r["keep"]):
        r["new_id"] = -1
OUT = ROOT / "runs" / "taxonomy_v3"
OUT.mkdir(exist_ok=True)
with open(OUT / "freeze.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["old_id", "name", "instances", "keep", "new_id"])
    for r in sorted(rows, key=lambda r: int(r["old_id"])):
        w.writerow([r["old_id"], r["name"], r["instances"], r["keep"], r["new_id"]])
dropped = sorted([r for r in rows if not int(r["keep"])], key=lambda r: int(r["old_id"]))
(OUT / "notes.md").write_text(
    "# Taxonomy freeze v3 (2026-09-08)\n\n"
    f"- v2 minus `unknown` (near-total background miss, pure noise class)\n"
    f"- kept: {len(kept)} / dropped: {len(rows) - len(kept)} -> **68 classes**\n"
    "- paired with copy_paste 0.3 round targeting +0.1 mAP50\n"
)
print(f"v3: kept={len(kept)} dropped={len(rows) - len(kept)}")
