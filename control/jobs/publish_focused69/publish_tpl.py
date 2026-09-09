"""Publish job: build the focused69 tree with REAL files and publish it as a
Kaggle dataset (persistent storage). Run ONCE; all future train runs attach
the published dataset instead of rebuilding.

- Copies (not symlinks) images, rewrites remapped labels, writes data.yaml
  + dataset-metadata.json, then dataset_create_new (private).
- Aborts if images exceed 14GB (/kaggle/working is ~20GB).
- Prints are flushed for the live log.

Freeze mapping is embedded by build_pub.py (same single-file constraint:
Kaggle uploads only code_file).
"""
import csv
import io
import base64
import os
import shutil
import sys
from pathlib import Path

FREEZE_CSV = base64.b64decode("__FREEZE_B64__").decode()

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "focused69_ds"
EXCLUDE_PREFIXES = ("schematic_images.hf.yolov8",)
DATASET_ID = "muzaafnameerfirdausi/circuitvision-focused69"
MAX_IMG_GB = 14


def log(msg):
    print(msg, flush=True)


def find_root():
    roots = []
    for cur, dirs, _ in os.walk(INPUT):
        if os.path.basename(cur) in ("images", "labels", "orientation_labels"):
            dirs[:] = []
            continue
        if "train" in dirs and (Path(cur) / "train" / "images").is_dir():
            roots.append(Path(cur))
            dirs[:] = [d for d in dirs if d not in ("train", "valid", "test")]
    assert roots, "no YOLO root"
    clahe = [r for r in roots if "clahe" in str(r).lower()]
    root = clahe[0] if clahe else roots[0]
    log(f"using input root: {root}")
    return root


def main():
    mapping, names = {}, {}
    for r in csv.DictReader(io.StringIO(FREEZE_CSV)):
        mapping[int(r["old_id"])] = int(r["new_id"])
        if int(r["keep"]):
            names[int(r["new_id"])] = r["name"]
    names = [names[i] for i in sorted(names)]
    log(f"freeze: {len(names)} kept classes")

    src_root = find_root()
    n_img = n_box = n_drop = n_excl = img_bytes = 0
    for split in ("train", "valid", "test"):
        sdir = src_root / split
        dimg, dlab = OUT / split / "images", OUT / split / "labels"
        dimg.mkdir(parents=True, exist_ok=True)
        dlab.mkdir(parents=True, exist_ok=True)
        if not (sdir / "images").is_dir():
            continue
        for img in sorted((sdir / "images").iterdir()):
            if img.name.lower().startswith(EXCLUDE_PREFIXES):
                n_excl += 1
                continue
            n_img += 1
            img_bytes += img.stat().st_size
            shutil.copy2(img, dimg / img.name)
            lf = sdir / "labels" / (img.stem + ".txt")
            out_lines = []
            if lf.is_file():
                for line in lf.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    new_id = mapping.get(int(float(parts[0])), -1)
                    if new_id < 0:
                        n_drop += 1
                        continue
                    out_lines.append(" ".join([str(new_id)] + parts[1:]))
                    n_box += 1
            (dlab / (img.stem + ".txt")).write_text(
                "\n".join(out_lines) + ("\n" if out_lines else ""))
            if n_img % 5000 == 0:
                log(f"  ...{n_img} images copied ({img_bytes / 1e9:.1f} GB)")
    log(f"copied: {n_img} images ({img_bytes / 1e9:.1f} GB), excl={n_excl}, boxes kept={n_box} dropped={n_drop}")
    assert img_bytes / 1e9 < MAX_IMG_GB, f"images {img_bytes / 1e9:.1f}GB exceed {MAX_IMG_GB}GB guard"

    (OUT / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\nnc: %d\nnames:\n%s\n"
        % (len(names), "".join(f"- {n}\n" for n in names)))
    import json
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "CircuitVision Focused69 YOLO", "id": DATASET_ID,
         "licenses": [{"name": "CC-BY-SA-3.0"}]}, indent=2))
    log("metadata written, uploading dataset (one-time, may take a while) ...")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    # NOTE: dir_mode="zip" is REQUIRED — default "skip" silently drops folders
    # (v1 published only data.yaml). Each split dir uploads as one archive.
    # Dataset shell already exists (v1) -> create_version; else create_new.
    try:
        r = api.dataset_create_version(str(OUT), "focused69 69-class tree",
                                       quiet=False, dir_mode="zip")
    except Exception as e:
        log(f"create_version failed ({str(e)[:120]}), trying create_new ...")
        r = api.dataset_create_new(str(OUT), public=False, quiet=False,
                                   dir_mode="zip")
    log(f"dataset create response: {r}")
    log("PUBLISH DONE")


if __name__ == "__main__":
    main()
