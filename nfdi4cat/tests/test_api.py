"""SPARQL API tests, using an injected in-memory graph (no disk artefact needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from transaminase_kg.api import create_app
from transaminase_kg.graph import build_graph


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(graph=build_graph()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.text == "ok"


def test_index_lists_examples(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["graph"]["reactions"] == 3
    assert "example_queries" in body


def test_select_all_reactions(client):
    query = (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "SELECT ?r WHERE { ?r a ta:Transamination }"
    )
    r = client.get("/sparql", params={"query": query})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/sparql-results+json")
    data = json.loads(r.text)
    assert len(data["results"]["bindings"]) == 3


def test_query_by_ec_number(client):
    query = (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "SELECT ?enz WHERE { ?enz ta:ecNumber '2.6.1.2' }"
    )
    r = client.get("/sparql", params={"query": query})
    bindings = json.loads(r.text)["results"]["bindings"]
    assert len(bindings) == 1


def test_cofactor_query_finds_all_three(client):
    query = (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
        "PREFIX obo: <http://purl.obolibrary.org/obo/> "
        "SELECT ?reaction WHERE { "
        "  ?reaction ta:catalyzedBy ?e . ?e ta:hasCofactor ?c . "
        "  ?c skos:exactMatch obo:CHEBI_18405 . }"
    )
    r = client.get("/sparql", params={"query": query})
    assert len(json.loads(r.text)["results"]["bindings"]) == 3


def test_csv_content_negotiation(client):
    query = (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "SELECT ?r WHERE { ?r a ta:Transamination }"
    )
    r = client.get("/sparql", params={"query": query}, headers={"accept": "text/csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")


def test_post_query(client):
    query = (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#> "
        "ASK { ?r a ta:Transamination }"
    )
    r = client.post("/sparql", data={"query": query})
    assert r.status_code == 200
    assert json.loads(r.text)["boolean"] is True


def test_malformed_query_returns_400(client):
    r = client.get("/sparql", params={"query": "SELECT ?x WHERE { this is not sparql"})
    assert r.status_code == 400


def test_empty_query_returns_422_or_400(client):
    # missing required param -> FastAPI 422; empty string -> our 400
    r = client.get("/sparql", params={"query": "  "})
    assert r.status_code in (400, 422)
