"""GPU probe: answer definitively what accelerator API-pushed kernels get."""
import os

import torch

print("cuda_available:", torch.cuda.is_available(), flush=True)
print("device_count:", torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print(f"gpu{i}:", torch.cuda.get_device_name(i), flush=True)
    try:
        p = torch.cuda.get_device_properties(i)
        print(f"  vram_GB: {p.total_memory / 1e9:.1f} sm: {p.major}.{p.minor}", flush=True)
    except Exception as e:
        print("  props error:", e, flush=True)
print("kaggle_env:", sorted(k for k in os.environ if "KAGGLE" in k), flush=True)
