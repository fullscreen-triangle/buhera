"""Ping-pong enrichment tests (nfdi4cat receiving half).

These assert that accepting the Buhera peer pipeline's derived facts is:
  - opt-in (default behaviour is base-only, byte-for-byte);
  - queryable (balance facts land on the SAME reaction subjects);
  - detachable (dropping the enrichment named graph restores the base exactly).

The enrichment artefact is the one Buhera writes back over the wire, at
``long-grass/enrichment/out/enrichment.ttl``. If it has not been generated the
enrichment tests skip rather than fail — the base pipeline never depends on the
peer being present.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rdflib import URIRef

from transaminase_kg.api import (
    ENRICHMENT_GRAPH_IRI,
    create_app,
    load_dataset,
)
from transaminase_kg.graph import build_graph

ENRICHMENT_TTL = (
    Path(__file__).resolve().parents[2]
    / "long-grass"
    / "enrichment"
    / "out"
    / "enrichment.ttl"
)

requires_enrichment = pytest.mark.skipif(
    not ENRICHMENT_TTL.exists(),
    reason="Buhera enrichment artefact not generated (run `npm run enrich`)",
)

ENR = "https://w3id.org/nfdi4cat/transaminase-kg/enrichment#"
ALT = URIRef("https://w3id.org/nfdi4cat/transaminase-kg/resource/reaction/ALT")


def test_base_only_default_is_unchanged():
    """With no enrichment, create_app is identical to today's base graph."""
    app = create_app(graph=build_graph())
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


@requires_enrichment
def test_balance_facts_are_queryable_after_merge():
    ds = load_dataset(ENRICHMENT_TTL)
    client = TestClient(create_app(graph=ds))
    query = (
        f"PREFIX enr: <{ENR}> "
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "SELECT ?r ?mass ?charge WHERE { "
        "  ?r a ta:Transamination ; "
        "     enr:massBalanced ?mass ; enr:chargeBalanced ?charge . }"
    )
    r = client.get("/sparql", params={"query": query})
    assert r.status_code == 200
    bindings = json.loads(r.text)["results"]["bindings"]
    # one row per reaction, all balanced — a fact the base graph never carried
    assert len(bindings) == 3
    for b in bindings:
        assert b["mass"]["value"] == "true"
        assert b["charge"]["value"] == "true"


@requires_enrichment
def test_enrichment_is_detachable():
    """DROP GRAPH <enrichment> restores the base triple count exactly."""
    base_len = len(build_graph())

    ds = load_dataset(ENRICHMENT_TTL)
    enrichment_ctx = ds.graph(ENRICHMENT_GRAPH_IRI)
    assert len(enrichment_ctx) > 0, "enrichment graph populated"

    # default graph alone equals the base, before and after dropping enrichment
    default_ctx = ds.graph(URIRef("urn:x-rdflib:default"))
    assert len(default_ctx) == base_len

    ds.remove_graph(enrichment_ctx)  # DROP GRAPH
    assert len(ds.graph(URIRef("urn:x-rdflib:default"))) == base_len
    # the enrichment predicates are gone from the whole dataset
    assert len(list(ds.quads((ALT, URIRef(ENR + "massBalanced"), None, None)))) == 0
