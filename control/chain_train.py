"""chain_train.py — crash-proof segmented training loop (the autosave thingy).

Problem: a Kaggle batch version that crashes/errors loses /kaggle/working.
Fix: train in chained versions. Every COMPLETED version persists outputs;
the next version attaches them as a kernel source and the driver seeds
last.pt to resume exactly where it died. Worst case loss = one segment.

Each segment inherits the checkpoint's total-epoch target (40): resume
continues toward 40, it does not add 40 more.

Usage:
    $env:KAGGLE_API_TOKEN = 'KGAT_xxx'
    python chain_train.py --tag focused69 --slug muzaafnameerfirdausi/cv-train-focused69-s1 --target 40 --max-iters 6

State: runs/chain_<tag>.json. Segment outputs: runs/<tag>_seg<N>/
Final weights: runs/<tag>_best/best.pt
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = HERE / "jobs" / "train_stage1"
FETCH_PATTERN = r".*\.log,.*\.pt,results\.csv,.*\.yaml,.*\.csv"


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.stdout:
        print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        raise SystemExit(f"command failed: {cmd[0]}")
    return r.stdout


def ctl(*args):
    return run([sys.executable, "kaggle_ctl.py", *args])


def poll_status(slug, interval=60, give_up_hours=11):
    import re
    deadline = time.time() + give_up_hours * 3600
    while time.time() < deadline:
        out = ctl("status", slug)
        m = re.search(r'"status":\s*"(\w+)"', out)
        st = m.group(1) if m else "?"
        print(f"{time.strftime('%H:%M:%S')} status={st}", flush=True)
        if st in ("COMPLETE", "ERROR") or "CANCEL" in st:
            return st
        time.sleep(interval)
    return "TIMEOUT"


def read_progress(outdir):
    """(epochs_done, best_map5095) from results.csv under downloaded outputs."""
    cands = sorted(outdir.rglob("results.csv"))
    if not cands:
        return 0, 0.0
    rows = list(csv.DictReader(cands[-1].open()))
    if not rows:
        return 0, 0.0
    last = rows[-1]
    ep = 0
    for k in ("epoch", " epoch"):
        if k in last and str(last[k]).strip().isdigit():
            ep = int(last[k])
            break
    fmap = 0.0
    for k in last:
        if "mAP50-95" in k:
            try:
                fmap = float(last[k])
            except ValueError:
                pass
    return ep, fmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="focused69")
    ap.add_argument("--slug", default="muzaafnameerfirdausi/cv-train-focused69-s1")
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--max-iters", type=int, default=6)
    a = ap.parse_args()

    state_path = HERE / "runs" / f"chain_{a.tag}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"iters": []}

    # kernel-metadata lives in the job folder; attach own outputs as seed source
    meta_path = JOB / "kernel-metadata.json"

    for it in range(len(state["iters"]), a.max_iters):
        seg = it + 1
        print(f"===== segment {seg} =====", flush=True)
        try:
            run([sys.executable, str(JOB / "build_job.py")], cwd=str(HERE))
            meta = json.loads(meta_path.read_text())
            if it > 0:
                meta["kernel_sources"] = [a.slug]
            meta_path.write_text(json.dumps(meta, indent=2))

            push_out = ctl("push", str(JOB.relative_to(HERE)))
            try:
                ver = json.loads(push_out[push_out.index("{"):])["versionNumber"]
            except Exception:
                ver = "?"
            print(f"pushed version {ver}", flush=True)

            st = poll_status(a.slug)
            outdir = HERE / "runs" / f"{a.tag}_seg{seg}"
            outdir.mkdir(parents=True, exist_ok=True)
            ctl("output", a.slug, str(outdir.relative_to(HERE)),
                "--pattern", FETCH_PATTERN)
            ep, fmap = read_progress(outdir)
            state["iters"].append({"seg": seg, "version": ver, "status": st,
                                   "epochs": ep, "map5095": fmap})
            state_path.write_text(json.dumps(state, indent=2))
            print(f"segment {seg}: status={st} epochs={ep} map50-95={fmap:.4f}", flush=True)

            if st == "COMPLETE" and ep >= a.target:
                best = sorted(outdir.rglob("best.pt"))
                fin = HERE / "runs" / f"{a.tag}_best"
                fin.mkdir(exist_ok=True)
                if best:
                    import shutil
                    shutil.copyfile(best[-1], fin / "best.pt")
                    print(f"PRESENTABLE MODEL -> {fin / 'best.pt'}", flush=True)
                print("TARGET REACHED", flush=True)
                return
            if st == "COMPLETE":
                print("version complete but short of target; resuming next segment", flush=True)
                continue
            # ERROR: resume only if a checkpoint survived in outputs
            if list(outdir.rglob("last.pt")) or list(outdir.rglob("best.pt")):
                print("error BUT checkpoint survived -> resuming next segment", flush=True)
                continue
            print("error with NO checkpoint; retrying segment once fresh", flush=True)
            # one fresh retry: drop kernel_sources so it restarts clean
            meta = json.loads(meta_path.read_text())
            meta.pop("kernel_sources", None)
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            # Never let one segment's infra failure (429, timeout, download
            # blowup) kill the whole chain: record and continue to next iter.
            print(f"segment {seg} infra failure: {str(e)[:300]}", flush=True)
            state["iters"].append({"seg": seg, "version": None,
                                   "status": "INFRA_FAIL", "epochs": 0,
                                   "map5095": 0.0, "error": str(e)[:300]})
            state_path.write_text(json.dumps(state, indent=2))
            time.sleep(300)

    print("MAX ITERS REACHED without target", flush=True)


if __name__ == "__main__":
    main()
