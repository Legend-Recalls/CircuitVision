"""GPU train job driver: link dataset -> apply taxonomy freeze -> train stage 1.

SELF-CONTAINED: Kaggle script pushes upload only code_file, so freeze.csv
and train_stage1.py are embedded below by build_job.py (placeholders
__FREEZE_CSV__ / __TRAIN_STAGE1_SRC__). Edit driver_tpl.py + inputs, then run
build_job.py to regenerate driver.py before pushing.

- Images are SYMLINKED (dataset is GBs; /kaggle/working is ~20GB), labels are
  rewritten remapped (dropped classes removed) into a parallel tree.
- Then delegates to the embedded train_stage1.py (resume-aware, save_period=1).
- Resume across versions: attach the previous version as a kernel source;
  its last.pt is auto-discovered under /kaggle/input and seeded.
- Prints are flushed so the Kaggle live log shows progress.
"""
import argparse
import base64
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

# base64 payloads (no quote collisions possible); filled by build_job.py
FREEZE_CSV = base64.b64decode("__FREEZE_B64__").decode()

TRAIN_STAGE1_SRC = base64.b64decode("__TRAIN_STAGE1_B64__").decode()

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
HERE = Path(__file__).resolve().parent

# Dataset decision (frozen 2026-09-08, amended):
# - OUT: schematic_images.hf 100k (off-mission bulk; user call)
# - IN: cghd + v3qwe + schematic.merged + ci2n + digitize_hcd (~13.6k train)
# CGHD exclusion was v1-only and is removed.
EXCLUDE_PREFIXES = ("schematic_images.hf.yolov8",)


def log(msg):
    print(msg, flush=True)


def ensure_ultralytics():
    # Kaggle's image does not guarantee ultralytics: install on first need.
    # (Needs enable_internet:true in kernel-metadata, already set.)
    try:
        import ultralytics  # noqa
        log(f"ultralytics {ultralytics.__version__} present")
        return
    except ImportError:
        pass
    log("installing ultralytics ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    assert r.returncode == 0, "pip install ultralytics failed"
    import ultralytics  # noqa
    log(f"ultralytics {ultralytics.__version__} installed")


def find_root():
    roots = []
    for cur, dirs, _ in os.walk(INPUT):
        if os.path.basename(cur) in ("images", "labels", "orientation_labels"):
            dirs[:] = []
            continue
        if "train" in dirs and (Path(cur) / "train" / "images").is_dir():
            roots.append(Path(cur))
            dirs[:] = [d for d in dirs if d not in ("train", "valid", "test")]
    assert roots, "no YOLO root under /kaggle/input"
    clahe = [r for r in roots if "clahe" in str(r).lower()]
    root = clahe[0] if clahe else roots[0]
    log(f"using input root: {root}")
    return root


def load_freeze():
    assert FREEZE_CSV.startswith("old_id,"), "driver.py not built: run build_job.py"
    mapping, names = {}, {}
    for r in csv.DictReader(io.StringIO(FREEZE_CSV)):
        mapping[int(r["old_id"])] = int(r["new_id"])
        if int(r["keep"]):
            names[int(r["new_id"])] = r["name"]
    log(f"freeze: {sum(1 for v in mapping.values() if v >= 0)} kept classes")
    return mapping, [names[i] for i in sorted(names)]


def materialize_train_script():
    assert "def main" in TRAIN_STAGE1_SRC, "driver.py not built: run build_job.py"
    # NOTE: /kaggle/src (HERE) is read-only on the box — materialize into WORK.
    p = WORK / "train_stage1.py"
    p.write_text(TRAIN_STAGE1_SRC)
    return p


def find_seed():
    # Previous version attached as kernel source -> its outputs land under
    # /kaggle/input/<slug>/...; look for the resume checkpoint.
    cands = sorted(INPUT.rglob("stage1_640_coarse/weights/last.pt"))
    return cands[-1] if cands else None


def build_short_tree(src_root, mapping, names, tag):
    dst_root = WORK / f"clahe_{tag}"
    if (dst_root / ".done").exists():
        log("short tree already built, reusing")
        return dst_root
    n_img, n_box, n_drop, n_excl = 0, 0, 0, 0
    for split in ("train", "valid", "test"):
        sdir, dimg, dlab = src_root / split, dst_root / split / "images", dst_root / split / "labels"
        dimg.mkdir(parents=True, exist_ok=True)
        dlab.mkdir(parents=True, exist_ok=True)
        if not (sdir / "images").is_dir():
            continue
        for img in sorted((sdir / "images").iterdir()):
            if img.name.lower().startswith(EXCLUDE_PREFIXES):
                n_excl += 1
                continue
            n_img += 1
            link = dimg / img.name
            if not link.exists():
                try:
                    link.symlink_to(img)
                except OSError:
                    pass
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
            (dlab / (img.stem + ".txt")).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            if n_img % 20000 == 0:
                log(f"  ...{n_img} images linked")
    (dst_root / ".done").write_text("ok")
    yaml_text = "train: train/images\nval: valid/images\ntest: test/images\n\nnc: %d\nnames:\n%s\n\npath: %s\n" % (
        len(names), "".join(f"- {n}\n" for n in names), dst_root)
    (dst_root / "data.yaml").write_text(yaml_text)
    log(f"short tree: {n_img} images linked, {n_excl} excluded ({EXCLUDE_PREFIXES}), {n_box} kept boxes, {n_drop} dropped boxes")
    return dst_root


def main():
    ap = argparse.ArgumentParser()
    # Batch/device are AUTO-SELECTED from visible GPUs unless overridden:
    # 2x GPU (T4x2 interactive) -> --device 0,1 --batch 32 (original recipe);
    # 1x GPU (API NvidiaTeslaT4) -> --device 0 --batch 16 (16GB-safe for s-p2).
    # Pass --device/--batch explicitly to override.
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="round68",
                    help="labels the linked tree + project dir for this dataset decision")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--copy-paste", type=float, default=0.3)
    ap.add_argument("--init-glob", default="best_stage2*.pt",
                    help="warm-start weights to find under /kaggle/input (empty=from scratch)")
    args = ap.parse_args()

    ensure_ultralytics()
    mapping, names = load_freeze()
    train_py = materialize_train_script()

    # Fail fast on CPU-only boxes (P100-with-broken-torch era); adapt batch.
    import torch

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(f"visible GPUs: {n_gpu}")
    assert n_gpu >= 1, "no GPU visible — refusing to burn hours on CPU"
    device = args.device or ("0,1" if n_gpu >= 2 else "0")
    batch = args.batch if args.batch is not None else (32 if n_gpu >= 2 else 16)
    log(f"device={device} batch={batch}")

    # Sanity: actually run one tiny CUDA op (P100+cu128 lies about availability).
    try:
        torch.zeros(1, device="cuda")
        log("cuda smoke test OK")
    except Exception as e:
        raise SystemExit(f"CUDA unusable despite visible GPU: {e}")

    src_root = find_root()
    dst_root = build_short_tree(src_root, mapping, names, args.tag)

    project = WORK / f"runs_{args.tag}"
    # Resume chain: previous version attached as kernel source -> seed last.pt.
    seed = find_seed()
    if seed is not None:
        dest = project / "stage1_640_coarse" / "weights" / "last.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(seed, dest)
        log(f"seeded resume checkpoint from {seed} ({seed.stat().st_size / 1e6:.1f} MB)")
    else:
        log("no seed checkpoint found")
    init = None
    if args.init_glob:
        cands = sorted(INPUT.rglob(args.init_glob))
        if cands:
            init = str(cands[-1])
            log(f"warm-start weights: {init}")
        else:
            log("no warm-start weights found, training from yolov8s.pt")

    cmd = [sys.executable, str(train_py),
           "--data", str(dst_root / "data.yaml"),
           "--project", str(project),
           "--epochs", str(args.epochs),
           "--batch", str(batch),
           "--device", device,
           "--copy-paste", str(args.copy_paste)]
    if init:
        cmd += ["--init", init]
    log("launching: " + " ".join(cmd))
    r = subprocess.run(cmd)
    log(f"train exit code: {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
