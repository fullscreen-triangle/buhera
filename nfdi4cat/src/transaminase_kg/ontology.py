"""The OWL ontology (T-Box) for transaminase reactions, built with Owlready2.

Design decisions, and why:

* **Small and closed.** A "toy" ontology as the brief asks — enough classes and
  properties to describe a PLP-dependent aminotransferase reaction and nothing
  more. Every class earns its place.

* **Reuse over invention (FAIR "I").** Compounds are typed against ChEBI and
  enzymes against the EC hierarchy via annotation properties (`skos:exactMatch`,
  a dedicated `ecNumber`), so an instance is anchored to the community
  identifiers rather than to strings we made up.

* **Reaction roles are properties, not classes.** Whether a compound is a
  substrate or a product is a *role in a reaction*, not an intrinsic type — the
  same L-glutamate is a product here and could be a substrate elsewhere. So
  `hasSubstrate` / `hasProduct` are object properties on `Reaction`, and the
  chemical identity lives once on the `ChemicalSpecies` individual.

* **Cofactor dependence is explicit.** `hasCofactor` with a value restriction
  records that all three enzymes are pyridoxal-phosphate proteins — a fact a
  reasoner can use and a chemist expects to see.

The module returns an Owlready2 ``Ontology`` object; serialisation to RDF/XML
is the caller's concern (see :mod:`transaminase_kg.build`).
"""

from __future__ import annotations

from owlready2 import (
    AnnotationProperty,
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    Thing,
    get_ontology,
)

from .reference import ONTOLOGY_IRI


def build_ontology():
    """Construct and return the transaminase T-Box as an Owlready2 ontology."""
    onto = get_ontology(ONTOLOGY_IRI + "#")

    with onto:
        # --- annotation properties (external anchoring) ---------------------
        class ecNumber(AnnotationProperty):
            """The Enzyme Commission number of an enzyme/reaction, e.g. '2.6.1.2'."""

        class keggReaction(AnnotationProperty):
            """KEGG REACTION identifier, e.g. 'R00258'."""

        class systematicName(AnnotationProperty):
            """The IUBMB systematic enzyme name."""

        # --- core classes ---------------------------------------------------
        class ChemicalSpecies(Thing):
            """A chemical entity participating in a reaction.

            Individuals are anchored to ChEBI via ``exactMatch`` (see graph.py)."""

        class Enzyme(Thing):
            """A biological catalyst, identified by its EC number."""

        class Cofactor(ChemicalSpecies):
            """A non-protein chemical required for catalysis (here: PLP)."""

        class Reaction(Thing):
            """A chemical transformation catalysed by an enzyme."""

        class Transamination(Reaction):
            """A reaction transferring an amino group between a donor and an acceptor.

            All instances in this KG are PLP-dependent aminotransferase reactions."""

        # --- object properties (reaction roles) -----------------------------
        class hasSubstrate(ObjectProperty):
            domain = [Reaction]
            range = [ChemicalSpecies]

        class hasProduct(ObjectProperty):
            domain = [Reaction]
            range = [ChemicalSpecies]

        class catalyzedBy(ObjectProperty):
            domain = [Reaction]
            range = [Enzyme]

        class catalyzes(ObjectProperty):
            """Inverse of catalyzedBy — an enzyme catalyses a reaction."""

            domain = [Enzyme]
            range = [Reaction]
            inverse_property = catalyzedBy

        class hasCofactor(ObjectProperty):
            domain = [Enzyme]
            range = [Cofactor]

        # Fine-grained transamination roles, refining the generic ones so a
        # reasoner still sees donor/acceptor as substrates.
        class hasAminoDonor(hasSubstrate):
            domain = [Transamination]
            range = [ChemicalSpecies]

        class hasAminoAcceptor(hasSubstrate):
            domain = [Transamination]
            range = [ChemicalSpecies]

        # --- data properties ------------------------------------------------
        class reversible(DataProperty, FunctionalProperty):
            domain = [Reaction]
            range = [bool]

        class mechanism(DataProperty, FunctionalProperty):
            domain = [Reaction]
            range = [str]

        # --- axioms a reasoner can act on -----------------------------------
        # Every Transamination is catalysed by an enzyme that depends on a cofactor.
        Transamination.is_a.append(catalyzedBy.some(Enzyme))
        # PLP is the cofactor across this KG: assert it at the class level so an
        # enzyme with no other cofactor is still known to need one.
        Enzyme.is_a.append(hasCofactor.some(Cofactor))

    return onto
