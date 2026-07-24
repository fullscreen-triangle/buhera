"""Scientific correctness tests — the reactions must be real chemistry.

A knowledge graph that asserts an unbalanced reaction is worse than useless; it
is confidently wrong. These tests prove every reaction in the reference data
conserves mass (atom counts) and charge, and that the shared species are used
consistently.
"""

from collections import Counter

import pytest

from transaminase_kg.reference import (
    L_GLUTAMATE,
    OXOGLUTARATE,
    PLP,
    REACTIONS,
)


@pytest.mark.parametrize("reaction", REACTIONS, ids=lambda r: r.ec)
def test_reaction_is_mass_balanced(reaction):
    """Atoms in == atoms out for every element."""
    lhs: Counter = Counter()
    rhs: Counter = Counter()
    for c in reaction.substrates:
        lhs.update(c.formula)
    for c in reaction.products:
        rhs.update(c.formula)
    assert lhs == rhs, f"{reaction.ec} not mass-balanced: {lhs} != {rhs}"


@pytest.mark.parametrize("reaction", REACTIONS, ids=lambda r: r.ec)
def test_reaction_is_charge_balanced(reaction):
    """Net charge is conserved across the reaction."""
    lhs = sum(c.charge for c in reaction.substrates)
    rhs = sum(c.charge for c in reaction.products)
    assert lhs == rhs, f"{reaction.ec} not charge-balanced: {lhs} != {rhs}"


@pytest.mark.parametrize("reaction", REACTIONS, ids=lambda r: r.ec)
def test_shared_acceptor_and_product(reaction):
    """All three are 2-oxoglutarate -> L-glutamate aminotransferases."""
    assert reaction.amino_acceptor is OXOGLUTARATE
    assert reaction.amino_product is L_GLUTAMATE


@pytest.mark.parametrize("reaction", REACTIONS, ids=lambda r: r.ec)
def test_cofactor_is_plp(reaction):
    """Every enzyme here is a pyridoxal-phosphate protein."""
    assert reaction.cofactor is PLP


def test_ec_numbers_are_the_three_specified():
    assert {r.ec for r in REACTIONS} == {"2.6.1.1", "2.6.1.2", "2.6.1.3"}


@pytest.mark.parametrize("reaction", REACTIONS, ids=lambda r: r.ec)
def test_every_species_has_external_identifiers(reaction):
    """FAIR: no species without a ChEBI and a KEGG anchor."""
    for c in (*reaction.substrates, *reaction.products, reaction.cofactor):
        assert c.chebi and c.chebi.isdigit(), f"{c.key} missing ChEBI"
        assert c.kegg.startswith("C"), f"{c.key} missing KEGG id"
