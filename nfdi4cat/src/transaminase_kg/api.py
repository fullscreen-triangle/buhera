"""A SPARQL-over-HTTP endpoint for the transaminase knowledge graph.

Implements the read subset of the **SPARQL 1.1 Protocol**:

    GET  /sparql?query=...          query via URL parameter
    POST /sparql   (form or body)   query via form field or raw body

Result serialisation is content-negotiated through the standard media types:
``application/sparql-results+json`` (default), ``text/csv``, and — for CONSTRUCT
/DESCRIBE — ``text/turtle``. A minimal ``/`` landing route lists example queries,
and FastAPI's OpenAPI docs are served at ``/docs``.

The graph is loaded once at startup from the Turtle artefact produced by
``build-kg``; if that artefact is missing it is built in-memory on the fly, so
the service is always runnable.
"""

from __future__ import annotations

from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from rdflib import Graph

from . import __version__
from .build import DEFAULT_OUT, TTL_FILENAME
from .graph import build_graph, graph_stats

# --- media types (SPARQL 1.1) -----------------------------------------------
SPARQL_JSON = "application/sparql-results+json"
SPARQL_CSV = "text/csv"
TURTLE = "text/turtle"

EXAMPLE_QUERIES: dict[str, str] = {
    "all reactions": (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "SELECT ?reaction ?label WHERE {\n"
        "  ?reaction a ta:Transamination ; rdfs:label ?label .\n"
        "}"
    ),
    "substrates and products of alanine transaminase": (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX res: <https://w3id.org/nfdi4cat/transaminase-kg/resource/>\n"
        "SELECT ?role ?name WHERE {\n"
        "  res:reaction/ALT ta:hasSubstrate ?s . ?s rdfs:label ?name .\n"
        "  BIND('substrate' AS ?role)\n"
        "} "
    ),
    "enzymes and their EC numbers": (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>\n"
        "SELECT ?enzyme ?ec WHERE { ?enzyme ta:ecNumber ?ec . }"
    ),
    "every reaction using pyridoxal 5'-phosphate": (
        "PREFIX ta: <https://w3id.org/nfdi4cat/transaminase-kg/ontology#>\n"
        "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
        "PREFIX obo: <http://purl.obolibrary.org/obo/>\n"
        "SELECT ?reaction WHERE {\n"
        "  ?reaction ta:catalyzedBy ?enz .\n"
        "  ?enz ta:hasCofactor ?cof .\n"
        "  ?cof skos:exactMatch obo:CHEBI_18405 .\n"
        "}"
    ),
}


def load_graph() -> Graph:
    """Load the built Turtle artefact, or build in-memory if it is absent."""
    ttl_path = DEFAULT_OUT / TTL_FILENAME
    if ttl_path.exists():
        g = Graph()
        g.parse(ttl_path, format="turtle")
        return g
    return build_graph()


def create_app(graph: Graph | None = None) -> FastAPI:
    """Application factory — injectable graph makes the app trivially testable."""
    g = graph if graph is not None else load_graph()

    app = FastAPI(
        title="Transaminase Knowledge Graph — SPARQL endpoint",
        version=__version__,
        description=(
            "A FAIR knowledge graph of three PLP-dependent transaminase reactions "
            "(EC 2.6.1.1/.2/.3), queryable via the SPARQL 1.1 protocol."
        ),
    )

    def run_query(query: str, accept: str) -> Response:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="empty SPARQL query")
        try:
            result = g.query(query)
        except Exception as exc:  # malformed query -> 400, not 500
            raise HTTPException(status_code=400, detail=f"SPARQL error: {exc}") from exc

        # CONSTRUCT / DESCRIBE return a graph
        if result.type in ("CONSTRUCT", "DESCRIBE"):
            return Response(result.serialize(format="turtle"), media_type=TURTLE)

        # ASK / SELECT: honour Accept header
        if SPARQL_CSV in accept:
            return Response(result.serialize(format="csv").decode(), media_type=SPARQL_CSV)
        # default: SPARQL results JSON
        body = result.serialize(format="json")
        if isinstance(body, bytes):
            body = body.decode()
        return Response(body, media_type=SPARQL_JSON)

    @app.get("/", response_class=JSONResponse, tags=["info"])
    def index() -> dict:
        """Service metadata and ready-to-run example queries."""
        return {
            "service": "transaminase-kg SPARQL endpoint",
            "version": __version__,
            "graph": graph_stats(g),
            "endpoints": {
                "sparql": "/sparql?query=<SPARQL>",
                "docs": "/docs",
                "health": "/health",
            },
            "example_queries": EXAMPLE_QUERIES,
        }

    @app.get("/health", response_class=PlainTextResponse, tags=["info"])
    def health() -> str:
        """Liveness probe: OK only if the graph is non-empty."""
        if len(g) == 0:
            raise HTTPException(status_code=503, detail="graph is empty")
        return "ok"

    @app.get("/sparql", tags=["sparql"])
    def sparql_get(
        request: Request,
        query: str = Query(..., description="A SPARQL 1.1 query string."),
    ) -> Response:
        return run_query(query, request.headers.get("accept", SPARQL_JSON))

    @app.post("/sparql", tags=["sparql"])
    async def sparql_post(request: Request, query: str | None = Form(default=None)) -> Response:
        # SPARQL protocol allows the query as a form field OR as a raw
        # application/sparql-query body.
        if query is None:
            body = (await request.body()).decode().strip()
            query = body or None
        if query is None:
            raise HTTPException(status_code=400, detail="no query provided")
        return run_query(query, request.headers.get("accept", SPARQL_JSON))

    return app


# module-level app for `uvicorn transaminase_kg.api:app`
app = create_app()
