from pathlib import Path

import pytest

from rhmoo.runner import run_experiment

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

pytestmark = pytest.mark.smoke


def test_run_experiment_writes_artifacts(tmp_path):
    out_dir = run_experiment(CONFIG_DIR / "smoke.yaml", results_root=tmp_path)
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "run_log.jsonl").exists()


def test_best_unique_molecules_spans_the_whole_run_not_the_final_population():
    """brief SS4 asks for "the top-k molecules per arm per seed". An elitist GA
    converges, so the best molecule found can be crowded out of the final
    population generations before the run ends.
    """
    import pandas as pd

    from rhmoo.runner import best_unique_molecules

    trajectory = pd.DataFrame(
        {
            "generation": [0, 0, 1, 2, 3],
            # The best molecule appears early and never again.
            "smiles": ["CCO", "c1ccccc1", "CCN", "CCO", "CCC"],
            "fitness": [0.9, 0.5, 0.6, 0.9, 0.4],
        }
    )
    top, n_unique = best_unique_molecules(trajectory, k=3)

    assert n_unique == 4
    assert top == ["CCO", "CCN", "c1ccccc1"]


def test_best_unique_molecules_keys_on_canonical_smiles():
    """SELFIES decoding emits whichever spelling the token sequence produces, so
    one molecule otherwise occupies several top-k slots.
    """
    import pandas as pd

    from rhmoo.runner import best_unique_molecules

    trajectory = pd.DataFrame(
        {
            "smiles": ["Cc1cc(Cl)ccc1N=C1NCCN1", "CC1=CC(Cl)=CC=C1N=C2NCCN2", "CCO"],
            "fitness": [0.8, 0.8, 0.3],
        }
    )
    top, n_unique = best_unique_molecules(trajectory, k=10)

    assert n_unique == 2, "the two spellings are the same molecule"
    assert len(top) == 2


def test_best_unique_molecules_drops_invalid_rows_without_dropping_valid_ones():
    import numpy as np
    import pandas as pd

    from rhmoo.runner import best_unique_molecules

    trajectory = pd.DataFrame(
        {
            "smiles": ["CCO", None, "CCN"],
            "fitness": [0.5, np.nan, 0.7],
        }
    )
    top, n_unique = best_unique_molecules(trajectory, k=10)
    assert top == ["CCN", "CCO"]
    assert n_unique == 2


def test_best_unique_molecules_is_order_independent():
    """Ties break on SMILES, not on evaluation order, so the selection is
    reproducible across machines (brief SS10).
    """
    import pandas as pd

    from rhmoo.runner import best_unique_molecules

    rows = pd.DataFrame(
        {"smiles": ["CCO", "CCN", "CCC", "CCCC"], "fitness": [0.5, 0.5, 0.5, 0.5]}
    )
    first, _ = best_unique_molecules(rows, k=3)
    second, _ = best_unique_molecules(rows.iloc[::-1].reset_index(drop=True), k=3)
    assert first == second


def test_ensemble_variance_record_distinguishes_unchecked_from_unavailable(tmp_path, monkeypatch):
    """brief SS3.3 wants a record of whether per-member predictions were
    available. The probe only runs when the model is loaded, so a run served
    entirely from cache has checked nothing -- reporting that as "unavailable"
    would be a false provenance claim rather than a missing one.
    """
    from rhmoo.predictors import ADMETPredictor

    predictor = ADMETPredictor(cache_path=None)
    assert predictor.ensemble_variance_available is None

    mapping = {
        True: "available",
        False: "unavailable",
        None: "not_probed_all_predictions_cached",
    }
    assert mapping[predictor.ensemble_variance_available] == "not_probed_all_predictions_cached"
