"""Inline freeze.csv into publish_tpl.py -> publish.py (single-file push)."""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
tpl = (HERE / "publish_tpl.py").read_text()
freeze_b64 = base64.b64encode((HERE / "freeze.csv").read_bytes()).decode()
assert "__FREEZE_B64__" in tpl
out_path = HERE / "publish.py"
out_path.write_text(tpl.replace("__FREEZE_B64__", freeze_b64))
print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

import py_compile
py_compile.compile(str(out_path), doraise=True)
print("compiles OK")
