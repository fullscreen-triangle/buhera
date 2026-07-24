"""Knowledge-graph integrity tests (the A-Box)."""

import pytest
from rdflib import RDF, RDFS, Graph, URIRef
from rdflib.namespace import SKOS

from transaminase_kg.graph import TA, build_graph, graph_stats
from transaminase_kg.reference import REACTIONS


@pytest.fixture(scope="module")
def g() -> Graph:
    return build_graph()


def test_graph_has_three_reactions(g):
    stats = graph_stats(g)
    assert stats["reactions"] == 3
    assert stats["enzymes"] == 3
    # 8 distinct species (3 donors + shared acceptor + 3 oxo products + glutamate)
    # plus the PLP cofactor = 9
    assert stats["species"] == 9


def test_every_reaction_has_two_substrates_two_products(g):
    for rxn in REACTIONS:
        r_uri = URIRef(
            f"https://w3id.org/nfdi4cat/transaminase-kg/resource/reaction/{rxn.key}"
        )
        substrates = set(g.objects(r_uri, TA.hasSubstrate))
        products = set(g.objects(r_uri, TA.hasProduct))
        assert len(substrates) == 2, f"{rxn.ec} substrate count"
        assert len(products) == 2, f"{rxn.ec} product count"


def test_every_species_links_to_chebi(g):
    species = set(g.subjects(RDF.type, TA.ChemicalSpecies)) | set(
        g.subjects(RDF.type, TA.Cofactor)
    )
    assert species, "no species in graph"
    for s in species:
        matches = list(g.objects(s, SKOS.exactMatch))
        assert any("CHEBI_" in str(m) for m in matches), f"{s} not linked to ChEBI"


def test_every_enzyme_depends_on_plp(g):
    plp = URIRef(
        "https://w3id.org/nfdi4cat/transaminase-kg/resource/species/pyridoxal-5-phosphate"
    )
    for enz in g.subjects(RDF.type, TA.Enzyme):
        assert (enz, TA.hasCofactor, plp) in g, f"{enz} missing PLP cofactor"


def test_all_entities_have_labels(g):
    typed = {s for s, _, _ in g.triples((None, RDF.type, None))}
    for s in typed:
        if isinstance(s, URIRef) and str(s).startswith(
            "https://w3id.org/nfdi4cat/transaminase-kg/resource/"
        ):
            assert list(g.objects(s, RDFS.label)), f"{s} has no rdfs:label"


def test_graph_roundtrips_through_turtle(g):
    """Serialising and reparsing must preserve every triple."""
    ttl = g.serialize(format="turtle")
    g2 = Graph()
    g2.parse(data=ttl, format="turtle")
    assert len(g2) == len(g)
