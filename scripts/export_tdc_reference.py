"""Export the predictor-training reference set from TDC.

Run in an isolated environment (see Makefile / README): PyTDC pins
rdkit<2024.3.1, which is incompatible with the admet-ai 2.x experiment
environment. Output is cached under data/reference/ and consumed as plain CSV,
so the experiment environment never imports TDC.

The reference set is the ADMET_Group benchmark `train_val` split for exactly the
endpoints used in the objective -- i.e. the molecules the objective's predictors
were fit on.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from tdc import BenchmarkGroup

ENDPOINTS = ["Caco2_Wang", "Solubility_AqSolDB", "hERG", "CYP3A4_Veith"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out_dir: Path, download_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    group = BenchmarkGroup(name="ADMET_Group", path=str(download_dir))
    manifest = {"source": "TDC ADMET_Group train_val split", "endpoints": {}}
    frames = []

    for name in ENDPOINTS:
        benchmark = group.get(name)
        train_val = benchmark["train_val"]
        path = out_dir / f"{name}_train_val.csv"
        train_val.to_csv(path, index=False)
        frames.append(train_val[["Drug"]].assign(endpoint=name))
        manifest["endpoints"][name] = {
            "n_molecules": int(len(train_val)),
            "columns": list(train_val.columns),
            "file": path.name,
            "sha256": sha256(path),
        }

    union = pd.concat(frames, ignore_index=True)
    union_smiles = sorted(set(union["Drug"].astype(str)))
    union_path = out_dir / "reference_union.csv"
    pd.DataFrame({"smiles": union_smiles}).to_csv(union_path, index=False)
    manifest["union"] = {
        "n_unique_molecules": len(union_smiles),
        "file": union_path.name,
        "sha256": sha256(union_path),
    }

    import importlib.metadata as md

    manifest["pytdc_version"] = md.version("PyTDC")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(root / "data" / "reference", root / "data" / "raw" / "tdc")
