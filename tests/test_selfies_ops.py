import random

import pytest

from rhmoo.selfies_ops import (
    ALPHABET,
    crossover,
    is_valid_smiles,
    mutate_delete,
    mutate_insert,
    mutate_substitute,
    random_mutation,
    smiles_to_tokens,
    tokens_to_smiles,
)

pytestmark = pytest.mark.smoke

MOLECULES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CNC(=O)c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccncc1",
    "CCOCC",
]


def test_tokenize_decode_matches_canonical_roundtrip():
    from rdkit import Chem

    for smiles in MOLECULES:
        tokens = smiles_to_tokens(smiles)
        decoded = tokens_to_smiles(tokens)
        assert Chem.CanonSmiles(decoded) == Chem.CanonSmiles(smiles)


@pytest.mark.parametrize("operator", [mutate_substitute, mutate_insert, mutate_delete])
def test_mutation_operators_always_decode_to_valid_molecule(operator):
    rng = random.Random(0)
    for smiles in MOLECULES:
        tokens = smiles_to_tokens(smiles)
        for _ in range(50):
            mutated = operator(tokens, rng)
            assert is_valid_smiles(tokens_to_smiles(mutated))


def test_random_mutation_always_decodes_to_valid_molecule():
    rng = random.Random(1)
    for smiles in MOLECULES:
        tokens = smiles_to_tokens(smiles)
        for _ in range(50):
            mutated, name = random_mutation(tokens, rng)
            assert name in {"substitute", "insert", "delete"}
            assert is_valid_smiles(tokens_to_smiles(mutated))


def test_mutation_operators_reproducible_under_fixed_seed():
    tokens = smiles_to_tokens("CC(=O)O")
    results_a = [mutate_substitute(tokens, random.Random(42)) for _ in range(5)]
    results_b = [mutate_substitute(tokens, random.Random(42)) for _ in range(5)]
    assert results_a == results_b


def test_crossover_always_decodes_to_valid_molecules():
    rng = random.Random(2)
    for a, b in zip(MOLECULES, MOLECULES[1:]):
        tokens_a, tokens_b = smiles_to_tokens(a), smiles_to_tokens(b)
        for _ in range(50):
            child_a, child_b = crossover(tokens_a, tokens_b, rng)
            assert is_valid_smiles(tokens_to_smiles(child_a))
            assert is_valid_smiles(tokens_to_smiles(child_b))


def test_crossover_falls_back_on_short_parents():
    child_a, child_b = crossover(["[C]"], ["[C]", "[C]"], random.Random(0))
    assert child_a == ["[C]"]
    assert child_b == ["[C]", "[C]"]


def test_alphabet_is_nonempty_and_deduplicated():
    assert len(ALPHABET) == len(set(ALPHABET))
    assert len(ALPHABET) > 0


def test_single_atom_substitution_can_yield_empty_molecule():
    """Known edge case, not a bug: substituting the sole token of a one-heavy-atom
    molecule with a ring/branch-only symbol has nothing to attach to and decodes to
    the empty molecule. Realistic (multi-token) molecules never hit this in practice
    (see test_mutation_operators_always_decode_to_valid_molecule); the GA loop treats
    it like any other invalid candidate -- logged and retried, never silently dropped
    (see test_ga.py::test_invalid_offspring_are_logged_not_silently_dropped).
    """
    tokens = smiles_to_tokens("C")
    mutated = mutate_substitute(tokens, random.Random(0))
    decoded = tokens_to_smiles(mutated)
    assert decoded == "" or is_valid_smiles(decoded)
    assert not is_valid_smiles("")
