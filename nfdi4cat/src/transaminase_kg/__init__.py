"""transaminase_kg — a FAIR knowledge graph of three transaminase reactions.

The package is deliberately small and layered:

    reference.py   verified scientific facts (KEGG / BRENDA / ChEBI), one source of truth
    ontology.py    the OWL T-Box (classes + properties) built with Owlready2
    graph.py       the RDF A-Box (instances for the three reactions) built with rdflib
    build.py       CLI: (re)generate the ontology + graph artefacts on disk
    api.py         a FastAPI SPARQL endpoint over the built graph

Nothing here invents chemistry. Every compound carries its ChEBI identifier and
every enzyme its EC number, so the graph is interoperable with the wider
catalysis-data ecosystem rather than a private silo.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("transaminase-kg")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
