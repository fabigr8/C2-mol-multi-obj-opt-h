"""SELFIES genetic algorithm (brief SS3.2): tournament selection, elitism,
mutation (+ optional crossover), full per-molecule trajectory logging -- every
evaluated molecule is recorded, not just survivors (brief: "the trajectory is
data").

Fitness (objective aggregation + arm-specific constraints) is injected via
`fitness_fn` so this module has no ADMET-AI dependency and stays fast to unit
test; the runner (Phase 9) wires in the real objective + constraint pipeline.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd

from rhmoo.config import GAConfig
from rhmoo.selfies_ops import (
    crossover as crossover_op,
    is_valid_smiles,
    random_mutation,
    smiles_to_tokens,
    tokens_to_smiles,
)

# Given a batch of SMILES, return one row per input (same order) with at least
# a "fitness" column; any other columns (component scores, penalties, ...)
# are propagated verbatim into the trajectory log.
FitnessFn = Callable[[list[str]], pd.DataFrame]


def _tournament_select(fitness: np.ndarray, k: int, rng: random.Random) -> int:
    contestants = rng.sample(range(len(fitness)), min(k, len(fitness)))
    return max(contestants, key=lambda i: fitness[i])


class SelfiesGA:
    def __init__(self, config: GAConfig, fitness_fn: FitnessFn, seed: int):
        self.config = config
        self.fitness_fn = fitness_fn
        self.rng = random.Random(seed)
        self.log_rows: list[dict] = []

    def _evaluate(
        self,
        smiles_list: list[str],
        generation: int,
        parents: list[tuple[str, ...]],
        operators: list[str],
    ) -> np.ndarray:
        result = self.fitness_fn(smiles_list)
        if len(result) != len(smiles_list):
            raise ValueError("fitness_fn must return exactly one row per input SMILES")
        for i, smiles in enumerate(smiles_list):
            row = result.iloc[i].to_dict()
            row.update(
                {
                    "generation": generation,
                    "smiles": smiles,
                    "parent_smiles": parents[i],
                    "operator": operators[i],
                }
            )
            self.log_rows.append(row)
        return result["fitness"].to_numpy(dtype=float)

    def _log_invalid(self, generation: int, parents: tuple[str, ...], operator: str) -> None:
        self.log_rows.append(
            {
                "generation": generation,
                "smiles": None,
                "parent_smiles": parents,
                "operator": operator,
                "fitness": np.nan,
                "valid": False,
            }
        )

    def _make_offspring(self, population: list[str], fitness: np.ndarray) -> tuple[str, tuple[str, ...], str]:
        parent_idx = _tournament_select(fitness, self.config.tournament_k, self.rng)
        parent_smiles = population[parent_idx]

        applied: list[str] = []
        if self.config.crossover_enabled and self.rng.random() < self.config.crossover_rate:
            partner_idx = _tournament_select(fitness, self.config.tournament_k, self.rng)
            partner_smiles = population[partner_idx]
            child_tokens, _ = crossover_op(
                smiles_to_tokens(parent_smiles), smiles_to_tokens(partner_smiles), self.rng
            )
            parents = (parent_smiles, partner_smiles)
            applied.append("crossover")
        else:
            child_tokens = smiles_to_tokens(parent_smiles)
            parents = (parent_smiles,)

        if self.rng.random() < self.config.mutation_rate:
            child_tokens, mutation_name = random_mutation(child_tokens, self.rng)
            applied.append(mutation_name)

        operator = "+".join(applied) if applied else "clone"
        return tokens_to_smiles(child_tokens), parents, operator

    def run(self, initial_population: list[str]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
        population = list(initial_population)
        fitness = self._evaluate(
            population, 0, [() for _ in population], ["init" for _ in population]
        )

        n_elite = max(1, int(len(population) * self.config.elitism_fraction))

        for generation in range(1, self.config.n_generations + 1):
            order = np.argsort(-fitness)
            elite_idx = order[:n_elite]
            elites = [population[i] for i in elite_idx]
            elite_fitness = fitness[elite_idx]

            n_offspring = len(population) - n_elite
            offspring: list[str] = []
            offspring_parents: list[tuple[str, ...]] = []
            offspring_operators: list[str] = []

            while len(offspring) < n_offspring:
                child_smiles, parents, operator = self._make_offspring(population, fitness)
                if not is_valid_smiles(child_smiles):
                    self._log_invalid(generation, parents, operator)
                    continue
                offspring.append(child_smiles)
                offspring_parents.append(parents)
                offspring_operators.append(operator)

            offspring_fitness = self._evaluate(offspring, generation, offspring_parents, offspring_operators)

            population = elites + offspring
            fitness = np.concatenate([elite_fitness, offspring_fitness])

        return population, fitness, pd.DataFrame(self.log_rows)
