"""kaggle_ctl.py — minimal CLI over the Kaggle API for CircuitVision training.

Token comes from $KAGGLE_API_TOKEN only. Nothing secret is written to disk.
All list/status/output calls retry with backoff (Kaggle 429s aggressively
after burst pagination — see hist download incident 2026-09-08).
"""
import argparse
import os
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

USER = "muzaafnameerfirdausi"


def api():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("Set $KAGGLE_API_TOKEN first (kaggle.com/settings -> Create New Token)")
    a = KaggleApi()
    a.authenticate()
    return a


def cmd_kernels(a, _):
    ks = a.kernels_list(user=USER)
    for k in ks:
        print(f"{k.ref} | {k.title} | v{k.current_version_number} | last_run={k.last_run_time}")


def cmd_datasets(a, _):
    for d in a.dataset_list(user=USER):
        print(f"{d.ref} | {d.title}")


def call_robust(fn, *args, tries=5, **kw):
    """Retry Kaggle API calls on 429/5xx/timeouts with exponential backoff."""
    delay = 30
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            msg = str(e)
            if i == tries - 1 or not any(
                    s in msg for s in ("429", "Too Many Requests", "500",
                                       "502", "503", "ConnectTimeout", "timed out")):
                raise
            print(f"  [retry] API busy ({msg[:80]}), waiting {delay}s...", flush=True)
            time.sleep(delay)
            delay *= 2


def cmd_status(a, args):
    print(call_robust(a.kernels_status, args.slug))


def cmd_push(a, args):
    r = a.kernels_push(args.folder)
    print(r)


def cmd_output(a, args):
    os.makedirs(args.out, exist_ok=True)
    # NOTE: kernels_output paginates (default 20/page). Always pass a pattern
    # so we fetch a handful of files, never the whole working tree.
    pats = args.pattern.split(",") if args.pattern else [None]
    n = 0
    for pat in pats:
        files, _ = call_robust(a.kernels_output, args.slug, args.out,
                               file_pattern=pat, force=True, quiet=True)
        n += len(files)
    print(f"downloaded {n} files -> {args.out} [{args.pattern}]")


def cmd_pull(a, args):
    os.makedirs(args.out, exist_ok=True)
    a.kernels_pull(args.slug, path=args.out)
    print(f"pulled {args.slug} -> {args.out}")


def cmd_watch(a, args):
    """Poll a run to completion, then fetch log/weights/metrics. No pushes."""
    import time
    while True:
        # NOTE: kernels_status returns a response OBJECT (not a string),
        # and status may be an enum int — match loosely, never regex-crash.
        out = call_robust(a.kernels_status, args.slug)
        s = str(out).upper()
        st = next((c for c in ("RUNNING", "QUEUED", "COMPLETE", "ERROR",
                               "CANCELLED", "CANCEL_ACKNOWLEDGED")
                   if c in s), "?")
        print(f"{time.strftime('%H:%M:%S')} status={st}", flush=True)
        if st in ("COMPLETE", "ERROR") or "CANCEL" in st:
            break
        time.sleep(60)
    os.makedirs(args.out, exist_ok=True)
    for pat in (r".*\.log", r".*\.pt", r"results\.csv", r".*\.yaml"):
        files, _ = call_robust(a.kernels_output, args.slug, args.out,
                               file_pattern=pat, force=True, quiet=True)
        print(f"fetched {len(files)} [{pat}]", flush=True)
    print(f"WATCH DONE status={st} -> {args.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("kernels")
    sub.add_parser("datasets")
    s = sub.add_parser("status")
    s.add_argument("slug")
    s = sub.add_parser("push")
    s.add_argument("folder")
    s = sub.add_parser("output")
    s.add_argument("slug")
    s.add_argument("out")
    s.add_argument("--pattern", default=None,
                   help="comma-separated REGEX patterns, e.g. '.*\\.log,.*\\.pt'")
    s = sub.add_parser("pull")
    s.add_argument("slug")
    s.add_argument("out")
    s = sub.add_parser("watch")
    s.add_argument("slug")
    s.add_argument("out")
    args = p.parse_args()
    a = api()
    {"kernels": cmd_kernels, "datasets": cmd_datasets, "status": cmd_status,
     "push": cmd_push, "output": cmd_output, "pull": cmd_pull,
     "watch": cmd_watch}[args.cmd](a, args)


if __name__ == "__main__":
    main()
