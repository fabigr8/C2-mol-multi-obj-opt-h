"""SELFIES-based mutation/crossover operators (brief SS3.2). Every operator
preserves syntactic validity by construction: SELFIES guarantees any sequence
of valid alphabet symbols decodes to some molecule, so 100% of offspring are
parseable -- this is unit-tested, not just asserted.
"""

from __future__ import annotations

import random

import selfies as sf
from rdkit import Chem

ALPHABET: tuple[str, ...] = tuple(sorted(sf.get_semantic_robust_alphabet()))


def smiles_to_tokens(smiles: str) -> list[str]:
    return list(sf.split_selfies(sf.encoder(smiles)))


def tokens_to_smiles(tokens: list[str]) -> str:
    return sf.decoder("".join(tokens))


def is_valid_smiles(smiles: str) -> bool:
    return smiles != "" and Chem.MolFromSmiles(smiles) is not None


def mutate_substitute(tokens: list[str], rng: random.Random) -> list[str]:
    if not tokens:
        return list(tokens)
    tokens = list(tokens)
    i = rng.randrange(len(tokens))
    tokens[i] = rng.choice(ALPHABET)
    return tokens


def mutate_insert(tokens: list[str], rng: random.Random) -> list[str]:
    tokens = list(tokens)
    i = rng.randrange(len(tokens) + 1)
    tokens.insert(i, rng.choice(ALPHABET))
    return tokens


def mutate_delete(tokens: list[str], rng: random.Random) -> list[str]:
    if len(tokens) <= 1:
        return list(tokens)
    tokens = list(tokens)
    del tokens[rng.randrange(len(tokens))]
    return tokens


MUTATION_OPERATORS = {
    "substitute": mutate_substitute,
    "insert": mutate_insert,
    "delete": mutate_delete,
}


def random_mutation(tokens: list[str], rng: random.Random) -> tuple[list[str], str]:
    name = rng.choice(sorted(MUTATION_OPERATORS))
    return MUTATION_OPERATORS[name](tokens, rng), name


def crossover(
    tokens_a: list[str], tokens_b: list[str], rng: random.Random
) -> tuple[list[str], list[str]]:
    """Single-point crossover; falls back to unchanged parents if either is too short."""
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return list(tokens_a), list(tokens_b)
    cut_a = rng.randrange(1, len(tokens_a))
    cut_b = rng.randrange(1, len(tokens_b))
    child_a = tokens_a[:cut_a] + tokens_b[cut_b:]
    child_b = tokens_b[:cut_b] + tokens_a[cut_a:]
    return child_a, child_b
