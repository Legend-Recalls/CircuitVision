"""Auth probe: can code running INSIDE a Kaggle kernel use the Kaggle API?

Read-only (zero side effects): authenticate with ambient env only, list own
datasets. If this works, kernels can publish datasets (persistent storage).
"""
import os

print("env has KAGGLE_API_V1_TOKEN:", bool(os.environ.get("KAGGLE_API_V1_TOKEN")), flush=True)
print("env has KAGGLE_API_TOKEN:", bool(os.environ.get("KAGGLE_API_TOKEN")), flush=True)

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("AUTHENTICATE OK", flush=True)
ds = api.dataset_list(user="muzaafnameerfirdausi")
print(f"datasets visible: {len(ds)}", flush=True)
for d in ds:
    print(" -", d.ref, flush=True)
