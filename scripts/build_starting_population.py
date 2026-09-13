"""Starting population (brief SS3.4): a drug-like subset sampled from a public
library, cached locally with a pinned source, checksum, filter criteria, and
sampling seed so re-running never silently fetches or samples different data.

Source: GuacaMol v1 training set (ChEMBL-derived, pre-filtered for
drug-likeness and reactive/undesirable groups by the GuacaMol curation
pipeline). See https://github.com/BenevolentAI/guacamol for the filter
criteria applied upstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from rdkit import Chem

GUACAMOL_TRAIN_URL = "https://ndownloader.figshare.com/files/13612760"
GUACAMOL_TRAIN_MD5 = "05ad85d871958a05c02ab51a4fde8530"
GUACAMOL_TRAIN_FILENAME = "guacamol_v1_train.smiles"

DEFAULT_N_SAMPLE = 10_000
DEFAULT_SEED = 0


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_guacamol_train(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / GUACAMOL_TRAIN_FILENAME
    if dest.exists() and _md5(dest) == GUACAMOL_TRAIN_MD5:
        return dest

    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(GUACAMOL_TRAIN_URL, tmp)
    digest = _md5(tmp)
    if digest != GUACAMOL_TRAIN_MD5:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"md5 mismatch downloading {GUACAMOL_TRAIN_URL}: expected {GUACAMOL_TRAIN_MD5}, got {digest}"
        )
    tmp.rename(dest)
    return dest


def build_starting_population(raw_path: Path, out_path: Path, n_sample: int, seed: int) -> dict:
    lines = [ln.strip() for ln in raw_path.read_text().splitlines() if ln.strip()]
    n_total = len(lines)

    order = list(range(n_total))
    random.Random(seed).shuffle(order)

    canonical: list[str] = []
    seen: set[str] = set()
    dropped_invalid = 0
    for i in order:
        if len(canonical) >= n_sample:
            break
        mol = Chem.MolFromSmiles(lines[i])
        if mol is None:
            dropped_invalid += 1
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon in seen:
            continue
        seen.add(canon)
        canonical.append(canon)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("smiles\n" + "\n".join(canonical) + "\n")

    manifest = {
        "source_url": GUACAMOL_TRAIN_URL,
        "source_md5": GUACAMOL_TRAIN_MD5,
        "source_version": "GuacaMol v1 training set",
        "source_file": raw_path.name,
        "n_total_pool": n_total,
        "n_sampled": len(canonical),
        "n_dropped_invalid_smiles": dropped_invalid,
        "sampling_seed": seed,
        "filter_criteria": (
            "GuacaMol v1 curation (ChEMBL-derived; MW/logP/HBD/HBA/TPSA/rotatable-bond "
            "bounds and reactive/undesirable-group filters applied upstream by GuacaMol). "
            "This step additionally deduplicates by canonical SMILES and drops any "
            "RDKit-unparseable entries."
        ),
        "output_file": out_path.name,
        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_path.parent / "starting_population_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/guacamol")
    parser.add_argument("--out", default="data/processed/starting_population.csv")
    parser.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    raw_path = download_guacamol_train(Path(args.raw_dir))
    manifest = build_starting_population(raw_path, Path(args.out), args.n_sample, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
