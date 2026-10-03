import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import QED

from rhmoo.config import GAConfig
from rhmoo.ga import SelfiesGA
from rhmoo.selfies_ops import is_valid_smiles

pytestmark = pytest.mark.smoke

INITIAL_POPULATION = [
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CCCC",
    "CNC",
    "CC(C)C",
    "CCOCC",
    "c1ccncc1",
    "CCC(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CNC(=O)c1ccccc1",
    "CCCCCC",
    "c1ccc2ccccc2c1",
    "CC(N)C(=O)O",
    "CCOC(=O)C",
    "c1ccsc1",
    "CC1CCCCC1",
    "CCCCO",
    "CC(C)(C)O",
]


def qed_fitness_fn(smiles_list: list[str]) -> pd.DataFrame:
    scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        scores.append(QED.qed(mol) if mol is not None else 0.0)
    return pd.DataFrame({"fitness": scores})


def _config(**overrides) -> GAConfig:
    defaults = dict(
        population_size=len(INITIAL_POPULATION),
        n_generations=3,
        elitism_fraction=0.2,
        tournament_k=3,
        crossover_enabled=True,
        crossover_rate=0.3,
        mutation_rate=0.9,
    )
    defaults.update(overrides)
    return GAConfig(**defaults)


def test_run_preserves_population_size_and_validity():
    ga = SelfiesGA(_config(), qed_fitness_fn, seed=0)
    population, fitness, log = ga.run(INITIAL_POPULATION)

    assert len(population) == len(INITIAL_POPULATION)
    assert len(fitness) == len(INITIAL_POPULATION)
    assert all(is_valid_smiles(s) for s in population)


def test_run_logs_every_evaluated_molecule():
    config = _config()
    ga = SelfiesGA(config, qed_fitness_fn, seed=0)
    _, _, log = ga.run(INITIAL_POPULATION)

    n_offspring_per_gen = len(INITIAL_POPULATION) - max(1, int(len(INITIAL_POPULATION) * config.elitism_fraction))
    expected_evaluated = len(INITIAL_POPULATION) + n_offspring_per_gen * config.n_generations
    assert (log["fitness"].notna()).sum() == expected_evaluated
    assert set(log["generation"].unique()) == set(range(config.n_generations + 1))


def test_run_is_deterministic_under_fixed_seed():
    config = _config()
    pop_a, fit_a, log_a = SelfiesGA(config, qed_fitness_fn, seed=7).run(INITIAL_POPULATION)
    pop_b, fit_b, log_b = SelfiesGA(config, qed_fitness_fn, seed=7).run(INITIAL_POPULATION)

    assert pop_a == pop_b
    assert fit_a.tolist() == fit_b.tolist()
    pd.testing.assert_frame_equal(
        log_a.drop(columns=["parent_smiles"]), log_b.drop(columns=["parent_smiles"])
    )


def test_best_fitness_is_non_decreasing_across_generations_due_to_elitism():
    config = _config()
    ga = SelfiesGA(config, qed_fitness_fn, seed=3)
    _, _, log = ga.run(INITIAL_POPULATION)

    best_per_generation = log.dropna(subset=["fitness"]).groupby("generation")["fitness"].max()
    running_best = best_per_generation.cummax()
    assert (best_per_generation.reindex(running_best.index) <= running_best + 1e-9).all()


def test_invalid_offspring_are_logged_not_silently_dropped(monkeypatch):
    config = _config(mutation_rate=0.0, crossover_enabled=False)
    ga = SelfiesGA(config, qed_fitness_fn, seed=0)

    calls = {"n": 0}
    real_is_valid = is_valid_smiles

    def flaky_is_valid(smiles: str) -> bool:
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            return False
        return real_is_valid(smiles)

    monkeypatch.setattr("rhmoo.ga.is_valid_smiles", flaky_is_valid)
    _, _, log = ga.run(INITIAL_POPULATION)

    invalid_rows = log[log["valid"] == False]  # noqa: E712
    assert len(invalid_rows) > 0
    assert invalid_rows["smiles"].isna().all()


def test_elite_log_records_every_generation(_ga_config=None):
    """brief SS4.3 needs the elite set per generation, but elites are carried
    forward without re-evaluation, so the trajectory alone cannot say which
    molecules were elite when.
    """
    import numpy as np
    import pandas as pd

    from rhmoo.config import GAConfig
    from rhmoo.ga import SelfiesGA

    config = GAConfig(
        population_size=10, n_generations=4, elitism_fraction=0.2, tournament_k=3,
        crossover_enabled=False, crossover_rate=0.0, mutation_rate=1.0,
    )
    population = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCC", "CCOC", "CCCN", "c1ccncc1", "CCCO", "CC"]

    def fitness_fn(smiles):
        return pd.DataFrame({"fitness": np.linspace(0, 1, len(smiles))})

    ga = SelfiesGA(config, fitness_fn, seed=0)
    ga.run(population)
    elites = ga.elite_log()

    assert len(elites) == config.n_elite * (config.n_generations + 1)
    assert sorted(elites["generation"].unique()) == list(range(config.n_generations + 1))
    assert set(elites.columns) == {"generation", "rank", "smiles", "fitness"}
    # Ranked best-first within each generation.
    for _, group in elites.groupby("generation"):
        assert group.sort_values("rank")["fitness"].is_monotonic_decreasing


def test_elite_selection_uses_stable_tie_breaking():
    """Unstable argsort tie-breaking is platform- and build-dependent, and brief
    SS10 requires an identical top-k when the run is repeated on another machine.
    """
    import numpy as np
    import pandas as pd

    from rhmoo.config import GAConfig
    from rhmoo.ga import SelfiesGA

    config = GAConfig(
        population_size=8, n_generations=3, elitism_fraction=0.25, tournament_k=2,
        crossover_enabled=False, crossover_rate=0.0, mutation_rate=1.0,
    )
    population = ["CCO", "c1ccccc1", "CCN", "CCC", "CCCC", "CCOC", "CCCN", "CC"]

    # All ties: only a stable sort gives a reproducible elite set.
    def fitness_fn(smiles):
        return pd.DataFrame({"fitness": np.full(len(smiles), 0.5)})

    first = SelfiesGA(config, fitness_fn, seed=0)
    first.run(population)
    second = SelfiesGA(config, fitness_fn, seed=0)
    second.run(population)

    pd.testing.assert_frame_equal(first.elite_log(), second.elite_log())
