# Transaminase Knowledge Graph

A small, FAIR knowledge graph of three transaminase reactions — modelled with an
OWL ontology, stored as RDF, and served over a SPARQL-over-HTTP API. Built for the
NFDI4Cat demo brief.

It is deliberately **small, focused, and functional**: a thin but complete
vertical slice of a research-data pipeline —
*define a vocabulary → describe real data with it → make it queryable → package it → automate the build.*

---

## What it models

Three pyridoxal-5′-phosphate (PLP)-dependent aminotransferases, each transferring
an amino group from an L-amino-acid donor onto the shared acceptor
**2-oxoglutarate**, yielding **L-glutamate** and the corresponding 2-oxo acid:

| EC | Enzyme | Reaction |
|---|---|---|
| [2.6.1.2](https://www.brenda-enzymes.org/enzyme.php?ecno=2.6.1.2) | alanine transaminase | L-alanine + 2-oxoglutarate ⇌ pyruvate + L-glutamate |
| [2.6.1.1](https://www.brenda-enzymes.org/enzyme.php?ecno=2.6.1.1) | aspartate transaminase | L-aspartate + 2-oxoglutarate ⇌ oxaloacetate + L-glutamate |
| [2.6.1.3](https://www.brenda-enzymes.org/enzyme.php?ecno=2.6.1.3) | cysteine transaminase | L-cysteine + 2-oxoglutarate ⇌ 2-oxo-3-sulfanylpropanoate + L-glutamate |

Every reaction is **mass- and charge-balanced** (proven in the test suite), and
every species and enzyme carries a stable external identifier (ChEBI, KEGG, EC).

---

## Scientific provenance — where the facts come from

Nothing here is hand-asserted chemistry. The reference data
([`reference.py`](src/transaminase_kg/reference.py)) is cross-checked against
authoritative databases, and each datum keeps its source identifier:

- **Reactions & EC numbers** — [KEGG ENZYME](https://www.genome.jp/kegg/) and [BRENDA](https://www.brenda-enzymes.org/).
- **Compound identities** — [ChEBI](https://www.ebi.ac.uk/chebi/), resolved via KEGG's `conv/chebi` cross-reference.
- **Cofactor** — pyridoxal 5′-phosphate, [ChEBI:18405](http://purl.obolibrary.org/obo/CHEBI_18405).
- **Compound formulae / charges** — ChEBI physiological (major-microspecies) forms, used to prove reaction balance.

Because every node is anchored to community identifiers, the graph is
**interoperable** — it extends the wider catalysis-data ecosystem rather than
forming a private silo. That is the "I" and "R" of FAIR.

---

## Design decisions (the *why*)

**Reaction roles are properties, not classes.** Whether a compound is a substrate
or a product is a *role in a particular reaction*, not an intrinsic type — the
same L-glutamate is a product in all three reactions and a substrate elsewhere in
metabolism. So chemical identity lives once on a `ChemicalSpecies` individual, and
`hasSubstrate` / `hasProduct` are object properties on the reaction. This keeps the
graph free of the duplication and contradiction that "SubstrateGlutamate" vs.
"ProductGlutamate" classes would create.

**T-Box and A-Box are separated but coherent.** The ontology (classes + properties)
is authored with **Owlready2** ([`ontology.py`](src/transaminase_kg/ontology.py));
the instance data is authored with **rdflib** ([`graph.py`](src/transaminase_kg/graph.py))
against the same IRIs. The API depends only on rdflib at query time — Owlready2 is a
build-time tool, not a runtime one.

**Fine-grained roles refine generic ones.** `hasAminoDonor` and `hasAminoAcceptor`
are sub-properties of `hasSubstrate`, so a specific transamination query and a
generic "what are the substrates" query both work. The generic role is also
asserted directly in the A-Box, so no reasoner is required at query time.

**One source of truth.** All facts live in `reference.py` as reviewed dataclasses;
the ontology and graph are pure transforms over that data. The science can be
audited in one place, independently of the RDF plumbing.

---

## Layout

```
nfdi4cat/
├── pyproject.toml              # uv-managed project + deps
├── Dockerfile                  # multi-stage build on the uv base image
├── docker-compose.yml
├── .gitlab-ci.yml              # test → build → push to the container registry
├── src/transaminase_kg/
│   ├── reference.py            # verified facts (KEGG / BRENDA / ChEBI) — one source of truth
│   ├── ontology.py             # OWL T-Box  (Owlready2)
│   ├── graph.py                # RDF A-Box  (rdflib)
│   ├── build.py                # CLI: materialise the ontology + graph artefacts
│   └── api.py                  # FastAPI SPARQL 1.1 endpoint
├── tests/                      # pytest: chemistry balance, graph integrity, SPARQL
└── queries/                    # example SPARQL + a smoke-test script
```

---

## Run it

### With uv (local)

```bash
uv sync --all-extras           # create the environment
uv run build-kg                # materialise ontology + graph artefacts
uv run pytest                  # 31 tests: chemistry, graph, API
uv run uvicorn transaminase_kg.api:app --reload
```

Then open <http://localhost:8000/> (service info + example queries) or
<http://localhost:8000/docs> (interactive OpenAPI).

### With Docker

```bash
docker compose up --build
# -> http://localhost:8000/
```

The graph is built **into the image**, so the container starts ready and needs no
network at runtime.

---

## Query it

The service implements the read subset of the **SPARQL 1.1 protocol**:

```bash
# via GET
curl --get http://localhost:8000/sparql --data-urlencode \
  'query=PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>
         SELECT ?r WHERE { ?r a ta:Transamination }'

# content negotiation: SPARQL-results JSON (default) or CSV
curl --get http://localhost:8000/sparql -H 'Accept: text/csv' \
  --data-urlencode 'query=...'
```

More examples in [`queries/example_queries.sparql`](queries/example_queries.sparql),
and a live smoke test in [`queries/smoke_test.sh`](queries/smoke_test.sh).

A query that traverses the ChEBI cross-reference — *"every reaction that depends on
pyridoxal 5′-phosphate"* — returns all three enzymes, demonstrating that the graph
is queryable **through community identifiers**, not just its own terms.

---

## Tests

```bash
uv run pytest -q
```

The suite is organised by concern:

- **`test_chemistry.py`** — every reaction is mass- and charge-balanced; the shared
  acceptor/product and PLP cofactor are consistent; every species has external IDs.
- **`test_graph.py`** — the A-Box has exactly 3 reactions / 3 enzymes / 9 species;
  every species links to ChEBI; every enzyme depends on PLP; the graph round-trips
  through Turtle losslessly.
- **`test_api.py`** — the SPARQL endpoint answers SELECT/ASK, negotiates JSON/CSV,
  and returns `400` on malformed queries.

All 31 tests pass on Python 3.11–3.12.

---

## CI/CD

[`.gitlab-ci.yml`](.gitlab-ci.yml) runs two stages on gitlab.com:

1. **test** — `ruff` lint + the pytest suite, on the uv image.
2. **build** — build the Docker image with Kaniko and push it to the project's
   GitLab Container Registry, tagged with the commit SHA (and `latest` on the
   default branch).

---

## License

MIT — see [LICENSE](LICENSE).
