"""Phase 0 spike 2: ADMET-AI configuration variants and throughput on this machine.

Not part of the experiment. Emits JSON to a file so Lightning's stdout progress
bars do not corrupt it.
"""

import contextlib
import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from admet_ai import ADMETModel
from admet_ai.constants import DEFAULT_DRUGBANK_PATH

warnings.filterwarnings("ignore")
for name in ("lightning", "lightning.pytorch", "lightning.pytorch.utilities"):
    logging.getLogger(name).setLevel(logging.ERROR)

OUT: dict = {"mps_available": torch.backends.mps.is_available()}

drugbank = pd.read_csv(DEFAULT_DRUGBANK_PATH)
OUT["drugbank_rows"] = len(drugbank)
OUT["drugbank_columns_sample"] = list(drugbank.columns[:12])

pool = drugbank["smiles"].dropna().astype(str).tolist()
OUT["pool_size"] = len(pool)


def quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def bench(label: str, n: int = 512, offset: int = 0, **kw):
    device = kw.pop("device", None)
    t0 = time.perf_counter()
    model = quiet(ADMETModel, **kw)
    load_s = time.perf_counter() - t0
    if device is not None:
        model.device = device
    chunk = pool[offset : offset + n]
    t0 = time.perf_counter()
    preds = quiet(model.predict, smiles=chunk)
    elapsed = time.perf_counter() - t0
    OUT.setdefault("variants", {})[label] = {
        "load_seconds": round(load_s, 2),
        "predict_seconds": round(elapsed, 3),
        "mols_per_sec": round(n / elapsed, 1),
        "n_columns": preds.shape[1],
        "n_rows": preds.shape[0],
    }
    return model, preds


bench("cpu_drugbank_physchem")
bench("cpu_nodrugbank_physchem", drugbank_path=None, offset=512)
bench("cpu_nodrugbank_nophyschem", drugbank_path=None, include_physchem=False, offset=1024)
if torch.backends.mps.is_available():
    bench("mps_nodrugbank_physchem", drugbank_path=None, device="mps", offset=1536)

# Does predict() silently drop invalid SMILES?
model, _ = bench("dropcheck", n=4, offset=0, drugbank_path=None)
mixed = [pool[0], "not_a_smiles", pool[1], "C(C(C"]
preds = quiet(model.predict, smiles=mixed)
OUT["invalid_handling"] = {
    "input_n": len(mixed),
    "output_n": len(preds),
    "output_index": list(preds.index),
    "silently_drops_invalid": len(preds) < len(mixed),
}

# Which columns correspond to the five objective components?
wanted = ["Solubility_AqSolDB", "hERG", "CYP3A4_Veith", "Caco2_Wang", "QED"]
OUT["objective_columns_present"] = {w: (w in preds.columns) for w in wanted}

out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/spike2.json")
out_path.write_text(json.dumps(OUT, indent=2, default=str))
