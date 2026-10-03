"""ADMET-AI + QED prediction, with a persistent cache keyed by canonical SMILES
(brief SS6: the GA will re-evaluate duplicates constantly; this is the main
lever on runtime cost).

`ADMETModel.predict()` silently drops invalid SMILES (verified in Phase 0:
4 in, 2 out, no exception) and can reorder/dedupe its output, so results are
always realigned onto the requested SMILES list rather than trusted as-is.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from rhmoo.cache import SmilesKeyedCache

ADMET_COMPONENTS = ("Solubility_AqSolDB", "hERG", "CYP3A4_Veith", "Caco2_Wang")

#: Per-molecule spread across each endpoint's 5-member Chemprop ensemble
#: (brief SS3.3/SS4.4). QED has no ensemble -- it is an exact RDKit computation,
#: not a learned prediction -- so it gets no spread column.
ENSEMBLE_STD_COLUMNS = tuple(f"{c}_ensemble_std" for c in ADMET_COMPONENTS)

OUTPUT_COLUMNS = (*ADMET_COMPONENTS, "QED")
CACHE_COLUMNS = (*OUTPUT_COLUMNS, *ENSEMBLE_STD_COLUMNS)

logger = logging.getLogger(__name__)


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_qed(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot compute QED for invalid SMILES: {smiles!r}")
    return QED.qed(mol)


@contextlib.contextmanager
def _quiet_admet_ai():
    """Lightning hardcodes a stdout progress bar; suppress it and its loggers."""
    warnings.filterwarnings("ignore")
    for name in ("lightning", "lightning.pytorch", "lightning.pytorch.utilities"):
        logging.getLogger(name).setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _admet_ai_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("admet-ai")
    except PackageNotFoundError:  # pragma: no cover - admet-ai is a pinned dependency
        return "unknown"


def predict_with_ensemble_spread(model, smiles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of `ADMETModel.predict` that keeps the ensemble spread instead of
    discarding it.

    Brief SS3.3 asks for ensemble prediction variance as a second,
    representation-matched reliability signal (the ECFP4 applicability domain
    does not describe where a Chemprop GNN is competent), and says to check
    whether ADMET-AI exposes per-member predictions before implementing
    anything custom.

    It does not expose them publicly, but it does compute them: its private
    `_make_ensemble_predictions` stacks all 5 members per ensemble and then
    averages them away. So the spread is free -- the same forward passes
    already run -- and this reimplements only the aggregation step.

    The cost is that we bypass `predict()` and therefore own the invalid-SMILES
    filtering and index realignment it does. Returns (mean, std) frames both
    indexed by the *surviving* SMILES, which the caller must reindex onto its
    own request; `tests/test_predictors.py` asserts the means agree with
    `predict()`.
    """
    import torch

    prepared, _ = model._prepare_smiles(smiles=smiles)
    mols, kept = model._filter_valid_molecules(smiles=prepared)
    data_loader = model._build_dataloader(mols=mols)

    import lightning.pytorch as pl

    task_to_mean: dict[str, object] = {}
    task_to_std: dict[str, object] = {}
    for tasks, models in zip(model.task_lists, model.model_lists):
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                accelerator=model.device, devices=1,
            )
            stacked = torch.stack(
                [torch.cat(trainer.predict(model=member, dataloaders=data_loader), dim=0) for member in models]
            )
            mean = stacked.mean(dim=0).numpy()
            # Population (biased) std over the 5 members, matching how the
            # mean is taken; with a fixed ensemble size the choice only
            # rescales the signal, but it has to be recorded either way.
            std = stacked.std(dim=0, unbiased=False).numpy()
        for i, task in enumerate(tasks):
            task_to_mean[task] = mean[:, i]
            task_to_std[task] = std[:, i]

    return (
        pd.DataFrame(task_to_mean, index=kept),
        pd.DataFrame(task_to_std, index=kept),
    )


def ensemble_spread_supported(model) -> bool:
    """Probe the private attributes `predict_with_ensemble_spread` relies on.

    admet-ai is pinned, but a pin is not a guarantee, and brief SS3.3 accepts
    "record that it was checked" as a valid outcome. So an unexpected shape
    degrades to NaN spread columns rather than failing the run.
    """
    try:
        if len(model.task_lists) != len(model.model_lists) or not model.model_lists:
            return False
        return all(len(members) > 1 for members in model.model_lists) and all(
            hasattr(model, attr) for attr in ("_prepare_smiles", "_filter_valid_molecules", "_build_dataloader")
        )
    except AttributeError:
        return False


class PredictionCache(SmilesKeyedCache):
    """The objective's prediction cache: ADMET-AI endpoints + QED, keyed by
    canonical SMILES. Provenance-tagged with the admet-ai version so a cache
    written by a different model release is rebuilt rather than reused.
    """

    def __init__(self, path: str | Path | None = None, columns: Iterable[str] = CACHE_COLUMNS):
        super().__init__(
            columns=columns,
            path=path,
            provenance={"producer": "admet-ai", "version": _admet_ai_version()},
        )


class ADMETPredictor:
    """Batched ADMET-AI + RDKit-QED prediction with an optional persistent cache."""

    def __init__(self, cache_path: str | Path | None = None):
        self.cache = PredictionCache(cache_path)
        self._model = None
        # None = not probed yet. Recorded in the run log either way, per brief SS3.3.
        self.ensemble_variance_available: bool | None = None

    def _load_model(self):
        if self._model is None:
            from admet_ai import ADMETModel

            with _quiet_admet_ai():
                self._model = ADMETModel(include_physchem=False, drugbank_path=None)
        return self._model

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        canonical = [canonicalize(s) for s in smiles]
        n_invalid = sum(c is None for c in canonical)
        if n_invalid:
            logger.warning("dropped %d unparseable SMILES out of %d requested", n_invalid, len(smiles))

        valid_canonical = [c for c in canonical if c is not None]
        to_query = self.cache.missing(valid_canonical)

        if to_query:
            model = self._load_model()
            if self.ensemble_variance_available is None:
                self.ensemble_variance_available = ensemble_spread_supported(model)
                if not self.ensemble_variance_available:
                    logger.warning(
                        "ADMET-AI does not expose per-ensemble-member predictions in the expected "
                        "shape; recording ensemble variance as unavailable (brief SS3.3)"
                    )

            spread = None
            if self.ensemble_variance_available:
                try:
                    with _quiet_admet_ai():
                        raw, spread = predict_with_ensemble_spread(model, to_query)
                except Exception:  # pragma: no cover - degrades rather than failing the run
                    logger.exception("ensemble-spread path failed; falling back to plain predict()")
                    self.ensemble_variance_available = False
                    spread = None
                    with _quiet_admet_ai():
                        raw = model.predict(smiles=to_query)
            else:
                with _quiet_admet_ai():
                    raw = model.predict(smiles=to_query)

            raw = raw.reindex(to_query)  # ADMETModel.predict can drop/reorder rows
            dropped_by_model = raw[list(ADMET_COMPONENTS)].isna().all(axis=1)
            if dropped_by_model.any():
                logger.warning(
                    "ADMET-AI dropped %d of %d queried SMILES", int(dropped_by_model.sum()), len(to_query)
                )
            rows = raw[list(ADMET_COMPONENTS)].copy()
            rows["QED"] = [compute_qed(s) for s in to_query]
            if spread is not None:
                spread = spread.reindex(to_query)
                for component in ADMET_COMPONENTS:
                    rows[f"{component}_ensemble_std"] = spread[component].to_numpy()
            else:
                for column in ENSEMBLE_STD_COLUMNS:
                    rows[column] = float("nan")
            self.cache.add(rows)
            self.cache.save()

        result = self.cache.lookup(canonical).reset_index(drop=True)
        result["input_smiles"] = smiles
        result["canonical_smiles"] = canonical
        result["valid"] = [c is not None for c in canonical]
        return result
