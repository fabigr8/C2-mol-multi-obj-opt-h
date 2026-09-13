"""Phase 0 spike: verify ADMET-AI on this machine and measure prediction throughput.

Not part of the experiment. Emits JSON to stdout.
"""

import inspect
import json
import time
from pathlib import Path

import pandas as pd
from admet_ai import ADMETModel

import admet_ai

OUT = {}

pkg_dir = Path(admet_ai.__file__).parent
OUT["package_dir"] = str(pkg_dir)
OUT["resource_files"] = sorted(str(p.relative_to(pkg_dir)) for p in pkg_dir.rglob("*.csv"))
OUT["admet_model_signature"] = str(inspect.signature(ADMETModel.__init__))
OUT["predict_signature"] = str(inspect.signature(ADMETModel.predict))

t0 = time.perf_counter()
model = ADMETModel()
OUT["model_load_seconds"] = round(time.perf_counter() - t0, 2)
OUT["model_attrs"] = [a for a in dir(model) if not a.startswith("_")]

# Reference molecules: prefer the DrugBank reference set shipped with the package.
smiles_pool: list[str] = []
for csv in pkg_dir.rglob("*.csv"):
    try:
        df = pd.read_csv(csv)
    except Exception:
        continue
    for col in df.columns:
        if col.lower() in {"smiles", "smiles_string"}:
            smiles_pool = df[col].dropna().astype(str).tolist()
            OUT["smiles_source"] = str(csv.relative_to(pkg_dir))
            break
    if smiles_pool:
        break

if not smiles_pool:
    OUT["smiles_source"] = "fallback_hardcoded"
    smiles_pool = [
        "O(c1ccc(cc1)CCOC)CC(O)CNC(C)C",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1CCC[C@H]1c1cccnc1",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Clc1ccccc1C1=NCC(=O)Nc2ccc(cc12)Cl",
    ] * 256

OUT["smiles_pool_size"] = len(smiles_pool)

preds = model.predict(smiles=smiles_pool[:8])
OUT["predict_return_type"] = type(preds).__name__
if isinstance(preds, pd.DataFrame):
    OUT["predict_columns"] = list(preds.columns)
    OUT["predict_head"] = json.loads(preds.head(2).to_json(orient="index"))

timings = {}
cursor = 0
for batch in (32, 128, 512):
    if cursor + batch > len(smiles_pool):
        cursor = 0
    chunk = smiles_pool[cursor : cursor + batch]
    cursor += batch
    t0 = time.perf_counter()
    model.predict(smiles=chunk)
    elapsed = time.perf_counter() - t0
    timings[batch] = {
        "seconds": round(elapsed, 3),
        "mols_per_sec": round(len(chunk) / elapsed, 1),
    }
OUT["throughput"] = timings

print(json.dumps(OUT, indent=2, default=str))
