"""The RDF knowledge graph (A-Box) for the three transaminase reactions.

Built with **rdflib** (the brief's triplestore). The graph instantiates the
classes and properties defined in :mod:`transaminase_kg.ontology` with the
verified data from :mod:`transaminase_kg.reference`.

Two FAIR commitments are honoured here:

* every compound individual carries ``skos:exactMatch`` to its ChEBI IRI and a
  ``skos:closeMatch`` to KEGG, so the node is resolvable in the wider ecosystem;
* every enzyme carries its EC number both as a literal and as an ``exactMatch``
  to the EC resolver IRI.

The result is a single, self-contained RDF graph that any SPARQL engine can
query — no dependency on Owlready2 at query time.
"""

from __future__ import annotations

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from .reference import BASE_IRI, ONTOLOGY_IRI, REACTIONS, Compound, Reaction, provenance

# Namespaces --------------------------------------------------------------
TA = Namespace(ONTOLOGY_IRI + "#")          # our ontology terms
RES = Namespace(BASE_IRI + "/resource/")    # our individuals (the A-Box)


def _species_uri(compound: Compound) -> URIRef:
    return RES[f"species/{compound.key}"]


def _enzyme_uri(reaction: Reaction) -> URIRef:
    return RES[f"enzyme/{reaction.ec}"]


def _reaction_uri(reaction: Reaction) -> URIRef:
    return RES[f"reaction/{reaction.key}"]


def _add_species(g: Graph, compound: Compound, *, is_cofactor: bool = False) -> URIRef:
    """Add a chemical-species individual once, idempotently, with its ChEBI/KEGG links."""
    uri = _species_uri(compound)
    if (uri, RDF.type, None) in g:  # already added
        return uri
    g.add((uri, RDF.type, TA.Cofactor if is_cofactor else TA.ChemicalSpecies))
    g.add((uri, RDFS.label, Literal(compound.label, lang="en")))
    # FAIR interoperability: anchor to community identifiers.
    g.add((uri, SKOS.exactMatch, URIRef(compound.chebi_iri)))
    g.add((uri, SKOS.closeMatch, URIRef(compound.kegg_iri)))
    return uri


def build_graph() -> Graph:
    """Construct and return the populated RDF knowledge graph."""
    g = Graph()
    g.bind("ta", TA)
    g.bind("res", RES)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("owl", OWL)

    # --- graph-level provenance (FAIR "R": reusable, sourced) ---------------
    dataset = URIRef(BASE_IRI)
    g.add((dataset, RDF.type, OWL.Ontology))
    g.add((dataset, OWL.imports, URIRef(ONTOLOGY_IRI)))
    g.add((dataset, DCTERMS.title,
           Literal("Transaminase reactions knowledge graph", lang="en")))
    g.add((dataset, DCTERMS.description,
           Literal("Three PLP-dependent aminotransferase reactions "
                   "(EC 2.6.1.1/.2/.3), sourced from KEGG, BRENDA and ChEBI.",
                   lang="en")))
    for src_key, src_val in provenance().items():
        g.add((dataset, DCTERMS.source, Literal(f"{src_key}: {src_val}")))

    # --- instances ----------------------------------------------------------
    for rxn in REACTIONS:
        rxn_uri = _reaction_uri(rxn)
        enz_uri = _enzyme_uri(rxn)

        # Enzyme
        g.add((enz_uri, RDF.type, TA.Enzyme))
        g.add((enz_uri, RDFS.label, Literal(rxn.recommended_name, lang="en")))
        g.add((enz_uri, TA.ecNumber, Literal(rxn.ec)))
        g.add((enz_uri, TA.systematicName, Literal(rxn.systematic_name, lang="en")))
        g.add((enz_uri, SKOS.exactMatch, URIRef(rxn.ec_iri)))

        # Cofactor (PLP) + dependence
        cof_uri = _add_species(g, rxn.cofactor, is_cofactor=True)
        g.add((enz_uri, TA.hasCofactor, cof_uri))

        # Reaction
        g.add((rxn_uri, RDF.type, TA.Transamination))
        g.add((rxn_uri, RDFS.label,
               Literal(f"{rxn.recommended_name} reaction", lang="en")))
        g.add((rxn_uri, TA.systematicName, Literal(rxn.systematic_name, lang="en")))
        g.add((rxn_uri, TA.keggReaction, Literal(rxn.kegg_reaction)))
        g.add((rxn_uri, SKOS.closeMatch, URIRef(rxn.kegg_reaction_iri)))
        g.add((rxn_uri, TA.reversible, Literal(rxn.reversible, datatype=XSD.boolean)))
        g.add((rxn_uri, TA.mechanism, Literal(rxn.mechanism, lang="en")))
        g.add((rxn_uri, TA.catalyzedBy, enz_uri))

        # Substrate roles: donor + acceptor
        donor_uri = _add_species(g, rxn.amino_donor)
        acceptor_uri = _add_species(g, rxn.amino_acceptor)
        g.add((rxn_uri, TA.hasAminoDonor, donor_uri))
        g.add((rxn_uri, TA.hasAminoAcceptor, acceptor_uri))
        # hasAminoDonor/Acceptor are sub-properties of hasSubstrate; assert the
        # generic role too so a plain `hasSubstrate` query still succeeds without
        # requiring a reasoner at query time.
        g.add((rxn_uri, TA.hasSubstrate, donor_uri))
        g.add((rxn_uri, TA.hasSubstrate, acceptor_uri))

        # Product roles
        for product in rxn.products:
            prod_uri = _add_species(g, product)
            g.add((rxn_uri, TA.hasProduct, prod_uri))

    return g


def graph_stats(g: Graph) -> dict[str, int]:
    """A small integrity summary, handy for the build log and tests."""
    def count(cls: URIRef) -> int:
        return len(set(g.subjects(RDF.type, cls)))

    return {
        "triples": len(g),
        "reactions": count(TA.Transamination),
        "enzymes": count(TA.Enzyme),
        "species": count(TA.ChemicalSpecies) + count(TA.Cofactor),
    }
