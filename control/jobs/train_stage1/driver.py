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
FREEZE_CSV = base64.b64decode("b2xkX2lkLG5hbWUsaW5zdGFuY2VzLGtlZXAsbmV3X2lkDQowLGFtcGVyaW1ldGVyLDg1LDAsLTENCjEsYW1wbGlmaWVyLDQyLDAsLTENCjIsYW1wbGlmaWVyLmRpZmZlcmVudGlhbCw0MSwwLC0xDQozLGFtcGxpZmllci5zaW5nbGVfZW5kLDY1MCwxLDANCjQsYW1wbGlmaWVyLnNpbmdsZV9pbnB1dF9zaW5nbGVfZW5kLDc4LDAsLTENCjUsYW5kLDExNjcsMSwxDQo2LGFudGVubmEsMTE5LDEsMg0KNyxibG9jayw0NDksMSwzDQo4LGNhcGFjaXRvciwxNDUyMCwxLDQNCjksY2FwYWNpdG9yLmFkanVzdGFibGUsMjY0LDEsNQ0KMTAsY2FwYWNpdG9yLnBvbGFyaXplZCwyNDczLDEsNg0KMTEsY2xvY2ssNTIsMCwtMQ0KMTIsY3Jvc3NvdmVyLDExNDE5LDEsNw0KMTMsY3Jvc3NvdmVyLmN1cnZlZCwxMDM3LDEsOA0KMTQsY3J5c3RhbCwxMzIsMSw5DQoxNSxjdXJyZW50X3NvdXJjZSwxODM5LDEsMTANCjE2LGN1cnJlbnRfc291cmNlLmFjLDQxNCwxLDExDQoxNyxjdXJyZW50X3NvdXJjZS5kYyw3OTIsMSwxMg0KMTgsZGVwZW5kZW50X2N1cnJlbnRfc291cmNlLDE0MSwxLDEzDQoxOSxkZXBlbmRlbnRfdm9sdGFnZV9zb3VyY2UsMzEsMCwtMQ0KMjAsZGZsaXBmbG9wLDAsMCwtMQ0KMjEsZGlhYyw0MCwwLC0xDQoyMixkaW9kZSw2ODkxLDEsMTQNCjIzLGRpb2RlLmxpZ2h0X2VtaXR0aW5nLDEzOTEsMSwxNQ0KMjQsZGlvZGUuemVuZXIsMTIxMSwxLDE2DQoyNSxkaXNwbGF5Ljdfc2VnbWVudCwzNSwwLC0xDQoyNixlbGVjdHJpY19iZWxsLDM3LDAsLTENCjI3LGV4cGxhbmF0b3J5LDYwOCwxLDE3DQoyOCxmdXNlLDM4OCwxLDE4DQoyOSxnbmQsMTU4NzQsMSwxOQ0KMzAsaGVhdGluZ19lbGVtZW50LDYwLDAsLTENCjMxLGluZHVjdG9yLDU4MTYsMSwyMA0KMzIsaW5kdWN0b3IuZmVycml0ZSwxNzMsMSwyMQ0KMzMsaW50ZWdyYXRlZF9jaXJjdWl0LDE0OTIsMSwyMg0KMzQsaW50ZWdyYXRlZF9jaXJjdWl0Lm5lNTU1LDI2MiwxLDIzDQozNSxpbnRlZ3JhdGVkX2NpcmN1aXQudm9sdGFnZV9yZWd1bGF0b3IsMzU4LDEsMjQNCjM2LGp1bmN0aW9uLDc4MTM1LDEsMjUNCjM3LGxhbXAsMzAxLDEsMjYNCjM4LGxpZ2h0LDE1MSwxLDI3DQozOSxtYWduZXRpYywzNjQsMSwyOA0KNDAsbWFnbmV0cm9uLDU2LDAsLTENCjQxLG1lY2hhbmljYWwsMzIsMCwtMQ0KNDIsbWljcm9waG9uZSw5MiwwLC0xDQo0Myxtb3NmZXQsMTY2MiwxLDI5DQo0NCxtb3RvciwxNzEsMSwzMA0KNDUsbmFuZCw0NzksMSwzMQ0KNDYsbm1vcywzNzU2LDEsMzINCjQ3LG5tb3MuYnVsayw0MzIsMSwzMw0KNDgsbm9yLDM5NiwxLDM0DQo0OSxub3QsODg1LDEsMzUNCjUwLG5wbiwzMTY4LDEsMzYNCjUxLG9wZXJhdGlvbmFsX2FtcGxpZmllciwxNjY5LDEsMzcNCjUyLG9wdGljYWwsMTI3LDEsMzgNCjUzLG9wdG9jb3VwbGVyLDE2MiwxLDM5DQo1NCxvciw2OTMsMSw0MA0KNTUscG1vcywzMTA2LDEsNDENCjU2LHBtb3MuYnVsaywzMTIsMSw0Mg0KNTcscG5wLDE4MjIsMSw0Mw0KNTgscG9ydCw3OTI5LDEsNDQNCjU5LHByb2JlLDE3MSwwLC0xDQo2MCxwcm9iZS5jdXJyZW50LDcyLDAsLTENCjYxLHByb2JlLnZvbHRhZ2UsMzIsMCwtMQ0KNjIscmVsYXksODksMCwtMQ0KNjMscmVzaXN0b3IsMzU0NjAsMSw0NQ0KNjQscmVzaXN0b3IuYWRqdXN0YWJsZSwxMTc1LDEsNDYNCjY1LHJlc2lzdG9yLnBob3RvLDI4NCwxLDQ3DQo2Nixzb2NrZXQsMTg0LDEsNDgNCjY3LHNwZWFrZXIsNDMwLDEsNDkNCjY4LHN3aXRjaCwzNDQyLDEsNTANCjY5LHRlcm1pbmFsLDExMDg3LDEsNTENCjcwLHRleHQsODk3NjUsMSw1Mg0KNzEsdGdhdGUsMCwwLC0xDQo3Mix0aHlyaXN0b3IsNDE1LDEsNTMNCjczLHRyYW5zZm9ybWVyLDY5NiwxLDU0DQo3NCx0cmFuc2lzdG9yLmJqdCw0ODQ5LDEsNTUNCjc1LHRyYW5zaXN0b3IucGhvdG8sMTc3LDEsNTYNCjc2LHRyaWFjLDk4LDAsLTENCjc3LHVua25vd24sMTkxMiwwLC0xDQo3OCx2LDQzOCwxLDU3DQo3OSx2YXJpc3RvciwxODQsMSw1OA0KODAsdmRkLDE1OTgsMSw1OQ0KODEsdm9sdGFnZSwxNTY0LDEsNjANCjgyLHZvbHRhZ2UuYWMsMTQwNCwxLDYxDQo4Myx2b2x0YWdlLmJhdHRlcnksMTIyNiwxLDYyDQo4NCx2b2x0YWdlLmRjLDMwNDYsMSw2Mw0KODUsdm9sdGFnZS5kYy5vbmVfcG9ydCw2NzQsMSw2NA0KODYsdm9sdGFnZS5saW5lcywxODcsMSw2NQ0KODcsdm9sdGltZXRlciw4MiwwLC0xDQo4OCx2c3MsMTMwOSwxLDY2DQo4OSx3YXR0aW1ldGVyLDM5LDAsLTENCjkwLHhub3IsNjksMCwtMQ0KOTEseG9yLDM2MywxLDY3DQo=").decode()

TRAIN_STAGE1_SRC = base64.b64decode("IiIiUGhhc2UgQSAvIFN0YWdlIDE6IGNvYXJzZSB0cmFpbmluZyBhdCA2NDAgb24geW9sb3Y4cy1wMiAoQ0xBSEUgZ3JheXNjYWxlIGRhdGEpLgoKU2Vzc2lvbi10aW1lb3V0IHNhZmU6IGNoZWNrcG9pbnRzIGV2ZXJ5IGVwb2NoOyBpZiBsYXN0LnB0IGV4aXN0cyB1bmRlcgotLXByb2plY3QsIHRoZSBydW4gYXV0by1yZXN1bWVzIGluc3RlYWQgb2YgcmVzdGFydGluZy4gUG9pbnQgLS1wcm9qZWN0IGF0IGEKcGVyc2lzdGVudCBsb2NhdGlvbiAoR29vZ2xlIERyaXZlIG9uIENvbGFiOyAva2FnZ2xlL3dvcmtpbmcgb24gS2FnZ2xlLCB0aGVuCnNhdmUgdGhlIG5vdGVib29rIHZlcnNpb24gdG8ga2VlcCBvdXRwdXRzKS4KCiAgICBweXRob24gdHJhaW5fc3RhZ2UxLnB5IC0tZGF0YSAva2FnZ2xlL2lucHV0LzxjbGFoZS1kcz4vZGF0YS55YW1sIFwKICAgICAgICAtLXByb2plY3QgL2thZ2dsZS93b3JraW5nL3J1bnNfcGhhc2VBCgpOb3RlcyB2cy4gdGhlIHJhdyBwbGFuOgotIGNhY2hlPVRydWUgaXMgT0ZGOiBLYWdnbGUgaW5wdXQgZGlycyBhcmUgcmVhZC1vbmx5IGFuZCBDb2xhYiBSQU0gY2FuJ3QgaG9sZAogIDExNWsgaW1hZ2VzLiB3b3JrZXJzPTQgSlBFRy1kZWNvZGluZyBncmF5c2NhbGUgaXMgZmFzdCBlbm91Z2guCi0geW9sb3Y4cy1wMiBoYXMgbm8gb2ZmaWNpYWwgLnB0LCBzbyB3ZSBidWlsZCBmcm9tIHlhbWwgYW5kIGxvYWQgeW9sb3Y4cy5wdAogIGJhY2tib25lIHdlaWdodHMgKHRyYW5zZmVyIGxlYXJuaW5nIGluc3RlYWQgb2YgZnJvbS1zY3JhdGNoKS4KLSBiYXRjaD0tMSA9IEF1dG9CYXRjaCAofjYwJSBWUkFNKS4gU2V0IGEgZml4ZWQgYmF0Y2ggaWYgdXNpbmcgMiBHUFVzIChERFAKICBkb2VzIG5vdCBzdXBwb3J0IEF1dG9CYXRjaCk6IC0tYmF0Y2ggMzIgLS1kZXZpY2UgMCwxCiIiIgppbXBvcnQgYXJncGFyc2UKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIHVsdHJhbHl0aWNzIGltcG9ydCBZT0xPCgpSVU5fTkFNRSA9ICJzdGFnZTFfNjQwX2NvYXJzZSIKCgpkZWYgbWFpbigpOgogICAgYXAgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tZGF0YSIsIHJlcXVpcmVkPVRydWUpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tcHJvamVjdCIsIGRlZmF1bHQ9InJ1bnNfcGhhc2VBIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1lcG9jaHMiLCB0eXBlPWludCwgZGVmYXVsdD00MCkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1iYXRjaCIsIHR5cGU9aW50LCBkZWZhdWx0PS0xKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWRldmljZSIsIGRlZmF1bHQ9Tm9uZSkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1pbml0IiwgZGVmYXVsdD1Ob25lLAogICAgICAgICAgICAgICAgICAgIGhlbHA9Indhcm0tc3RhcnQgd2VpZ2h0cyAoZS5nLiBwcmV2aW91cyBiZXN0LnB0KTsgIgogICAgICAgICAgICAgICAgICAgICAgICAgImJhY2tib25lIHRyYW5zZmVycywgaGVhZCByZXNldHMgb24gbmMgbWlzbWF0Y2giKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWNvcHktcGFzdGUiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTAuMCkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1wYXRpZW5jZSIsIHR5cGU9aW50LCBkZWZhdWx0PTE1KQogICAgYXJncyA9IGFwLnBhcnNlX2FyZ3MoKQoKICAgIGxhc3QgPSBQYXRoKGFyZ3MucHJvamVjdCkgLyBSVU5fTkFNRSAvICJ3ZWlnaHRzIiAvICJsYXN0LnB0IgogICAgaWYgbGFzdC5leGlzdHMoKToKICAgICAgICBwcmludChmIlJlc3VtaW5nIGZyb20ge2xhc3R9IikKICAgICAgICBtb2RlbCA9IFlPTE8oc3RyKGxhc3QpKQogICAgICAgIG1vZGVsLnRyYWluKHJlc3VtZT1UcnVlKQogICAgICAgIHJldHVybgoKICAgIGlmIGFyZ3MuaW5pdDoKICAgICAgICBwcmludChmIldhcm0gc3RhcnQgZnJvbSB7YXJncy5pbml0fSAobmMgbWlzbWF0Y2ggLT4gYmFja2JvbmUgdHJhbnNmZXIsIGZyZXNoIGhlYWQpIikKICAgICAgICBtb2RlbCA9IFlPTE8oYXJncy5pbml0KQogICAgZWxzZToKICAgICAgICBtb2RlbCA9IFlPTE8oInlvbG92OHMtcDIueWFtbCIpLmxvYWQoInlvbG92OHMucHQiKQogICAgbW9kZWwudHJhaW4oCiAgICAgICAgZGF0YT1hcmdzLmRhdGEsCiAgICAgICAgZXBvY2hzPWFyZ3MuZXBvY2hzLAogICAgICAgIGltZ3N6PTY0MCwKICAgICAgICBiYXRjaD1hcmdzLmJhdGNoLAogICAgICAgIGRldmljZT1hcmdzLmRldmljZSwKICAgICAgICB3b3JrZXJzPTQsCiAgICAgICAgb3B0aW1pemVyPSJBZGFtVyIsCiAgICAgICAgbHIwPTAuMDAxLAogICAgICAgIGNvc19scj1UcnVlLAogICAgICAgIGNsb3NlX21vc2FpYz0xMCwKICAgICAgICBjb3B5X3Bhc3RlPWFyZ3MuY29weV9wYXN0ZSwKICAgICAgICAjIE9yaWVudGF0aW9uIGlzIGNsYXNzLWRlZmluaW5nIChucG4gdnMgcG5wLCBubW9zIHZzIHBtb3MpOiBubyBmbGlwcy4KICAgICAgICBmbGlwbHI9MC4wLAogICAgICAgIGZsaXB1ZD0wLjAsCiAgICAgICAgZGVncmVlcz0wLjAsCiAgICAgICAgIyBHcmF5c2NhbGUgaW5wdXQ6IGh1ZS9zYXQgaml0dGVyIGlzIG1lYW5pbmdsZXNzLCBrZWVwIGJyaWdodG5lc3Mgb25seS4KICAgICAgICBoc3ZfaD0wLjAsCiAgICAgICAgaHN2X3M9MC4wLAogICAgICAgIGhzdl92PTAuNCwKICAgICAgICBwcm9qZWN0PWFyZ3MucHJvamVjdCwKICAgICAgICBuYW1lPVJVTl9OQU1FLAogICAgICAgIGV4aXN0X29rPVRydWUsCiAgICAgICAgc2F2ZV9wZXJpb2Q9MSwKICAgICAgICBwYXRpZW5jZT1hcmdzLnBhdGllbmNlLAogICAgKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK").decode()

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
