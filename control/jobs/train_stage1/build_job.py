"""Build the self-contained push artifact for the train job.

Kaggle script pushes upload ONLY code_file, so this inlines freeze.csv and
train_stage1.py (base64, collision-proof) into driver_tpl.py -> driver.py.
Run before every push:

    python build_job.py
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
tpl = (HERE / "driver_tpl.py").read_text()
freeze_b64 = base64.b64encode((HERE / "freeze.csv").read_bytes()).decode()
train_b64 = base64.b64encode((HERE / "train_stage1.py").read_bytes()).decode()

assert "__FREEZE_B64__" in tpl and "__TRAIN_STAGE1_B64__" in tpl
out = tpl.replace("__FREEZE_B64__", freeze_b64).replace("__TRAIN_STAGE1_B64__", train_b64)
out_path = HERE / "driver.py"
out_path.write_text(out)
print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

import py_compile
py_compile.compile(str(out_path), doraise=True)
print("compiles OK")

# sanity: payloads round-trip
ns = {}
exec(compile("import base64\n" + "\n".join(
    [l for l in out.splitlines() if l.startswith(("FREEZE_CSV", "TRAIN_STAGE1_SRC"))]), "<check>", "exec"), ns)
assert ns["FREEZE_CSV"].startswith("old_id,") and "def main" in ns["TRAIN_STAGE1_SRC"]
print("payloads round-trip OK")
