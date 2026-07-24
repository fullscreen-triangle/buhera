"""Build CLI: (re)generate the ontology and knowledge-graph artefacts on disk.

    build-kg [--out DIR]

Writes:
    <out>/transaminase.owl.rdf   the T-Box (OWL/XML, from Owlready2)
    <out>/transaminase.ttl       the A-Box (Turtle, from rdflib)

The API loads the Turtle artefact at startup, so this step is the single place
where the graph is materialised. Deterministic: same inputs -> same output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .graph import build_graph, graph_stats
from .ontology import build_ontology

DEFAULT_OUT = Path(__file__).resolve().parent / "data"
OWL_FILENAME = "transaminase.owl.rdf"
TTL_FILENAME = "transaminase.ttl"


def build_artifacts(out_dir: Path) -> dict[str, object]:
    """Write both artefacts and return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    onto = build_ontology()
    owl_path = out_dir / OWL_FILENAME
    onto.save(file=str(owl_path), format="rdfxml")

    g = build_graph()
    ttl_path = out_dir / TTL_FILENAME
    g.serialize(destination=str(ttl_path), format="turtle")

    return {
        "owl": str(owl_path),
        "ttl": str(ttl_path),
        "stats": graph_stats(g),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the transaminase KG artefacts.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    result = build_artifacts(args.out)
    stats = result["stats"]
    print(f"ontology -> {result['owl']}")
    print(f"graph    -> {result['ttl']}")
    print(
        "  {triples} triples: {reactions} reactions, "
        "{enzymes} enzymes, {species} species".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
