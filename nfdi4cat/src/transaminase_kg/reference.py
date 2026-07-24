"""Verified scientific reference data — the single source of truth.

Every fact below is cross-checked against authoritative databases and carries a
stable external identifier so downstream artefacts (ontology, graph) never
hard-code an unsourced value:

  * Reactions / EC numbers .... KEGG ENZYME + BRENDA
  * Compound identities ........ ChEBI (via KEGG `conv/chebi` cross-references)
  * Cofactor ................... pyridoxal 5'-phosphate (PLP), ChEBI:18405

All three enzymes are pyridoxal-phosphate-dependent aminotransferases operating a
ping-pong bi-bi mechanism; each transfers an amino group from an L-amino-acid
donor to the common acceptor 2-oxoglutarate, yielding L-glutamate plus the
corresponding 2-oxo acid.

Keeping this as plain dataclasses (not buried in the ontology code) means the
science can be reviewed on its own, and the ontology/graph builders are pure
transforms over reviewed data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- external vocabularies (FAIR: reuse, don't reinvent) --------------------
CHEBI = "http://purl.obolibrary.org/obo/CHEBI_"          # OBO PURL for ChEBI
KEGG_COMPOUND = "https://identifiers.org/kegg.compound:"  # identifiers.org resolver
KEGG_REACTION = "https://identifiers.org/kegg.reaction:"
EC_RESOLVER = "https://identifiers.org/ec-code:"


@dataclass(frozen=True)
class Compound:
    """A chemical species, identified by its ChEBI accession.

    ``formula`` and ``charge`` are the physiological (major microspecies) values
    from ChEBI and are used to prove each reaction is mass- and charge-balanced
    (see tests). They are element-count maps, e.g. ``{"C": 3, "H": 7, "N": 1,
    "O": 2}`` for L-alanine.
    """

    key: str            # local slug, e.g. "L-alanine"
    label: str          # human-readable name
    chebi: str          # ChEBI numeric accession, e.g. "16977"
    kegg: str           # KEGG COMPOUND id, e.g. "C00041"
    formula: dict[str, int] = field(default_factory=dict)
    charge: int = 0

    @property
    def chebi_iri(self) -> str:
        return f"{CHEBI}{self.chebi}"

    @property
    def kegg_iri(self) -> str:
        return f"{KEGG_COMPOUND}{self.kegg}"


@dataclass(frozen=True)
class Reaction:
    """A transaminase reaction: donor + acceptor <=> oxo-product + glutamate."""

    key: str                    # local slug, e.g. "ALT"
    ec: str                     # EC number, e.g. "2.6.1.2"
    recommended_name: str
    systematic_name: str
    kegg_reaction: str          # KEGG REACTION id, e.g. "R00258"
    amino_donor: Compound       # the L-amino-acid substrate
    amino_acceptor: Compound    # 2-oxoglutarate (shared)
    oxo_product: Compound       # the 2-oxo acid produced from the donor
    amino_product: Compound     # L-glutamate (shared)
    cofactor: Compound
    reversible: bool = True
    mechanism: str = "ping-pong bi-bi"

    @property
    def ec_iri(self) -> str:
        return f"{EC_RESOLVER}{self.ec}"

    @property
    def kegg_reaction_iri(self) -> str:
        return f"{KEGG_REACTION}{self.kegg_reaction}"

    @property
    def substrates(self) -> tuple[Compound, Compound]:
        return (self.amino_donor, self.amino_acceptor)

    @property
    def products(self) -> tuple[Compound, Compound]:
        return (self.oxo_product, self.amino_product)


# --- compounds (ChEBI + KEGG verified) --------------------------------------
# Formulae/charges are the ChEBI physiological (major-microspecies) forms; the
# zwitterionic amino acids are net-neutral, their conjugate acids/bases carry the
# charge ChEBI records. They must balance across each reaction (proven in tests).
L_ALANINE = Compound(
    "L-alanine", "L-alanine", "16977", "C00041",
    formula={"C": 3, "H": 7, "N": 1, "O": 2}, charge=0,
)
L_ASPARTATE = Compound(
    "L-aspartate", "L-aspartate", "17053", "C00049",
    formula={"C": 4, "H": 6, "N": 1, "O": 4}, charge=-1,
)
L_CYSTEINE = Compound(
    "L-cysteine", "L-cysteine", "17561", "C00097",
    formula={"C": 3, "H": 7, "N": 1, "O": 2, "S": 1}, charge=0,
)
OXOGLUTARATE = Compound(
    "2-oxoglutarate", "2-oxoglutarate", "16810", "C00026",
    formula={"C": 5, "H": 4, "O": 5}, charge=-2,
)

PYRUVATE = Compound(
    "pyruvate", "pyruvate", "15361", "C00022",
    formula={"C": 3, "H": 3, "O": 3}, charge=-1,
)
OXALOACETATE = Compound(
    "oxaloacetate", "oxaloacetate", "16452", "C00036",
    formula={"C": 4, "H": 2, "O": 5}, charge=-2,
)
# 2-oxo-3-sulfanylpropanoate — the deaminated product of L-cysteine
MERCAPTOPYRUVATE = Compound(
    "2-oxo-3-sulfanylpropanoate", "2-oxo-3-sulfanylpropanoate", "16208", "C00957",
    formula={"C": 3, "H": 3, "O": 3, "S": 1}, charge=-1,
)
L_GLUTAMATE = Compound(
    "L-glutamate", "L-glutamate", "16015", "C00025",
    formula={"C": 5, "H": 8, "N": 1, "O": 4}, charge=-1,
)

PLP = Compound(
    "pyridoxal-5-phosphate", "pyridoxal 5'-phosphate", "18405", "C00018",
    formula={"C": 8, "H": 8, "N": 1, "O": 6, "P": 1}, charge=-2,
)

COMPOUNDS: tuple[Compound, ...] = (
    L_ALANINE, L_ASPARTATE, L_CYSTEINE, OXOGLUTARATE,
    PYRUVATE, OXALOACETATE, MERCAPTOPYRUVATE, L_GLUTAMATE, PLP,
)

# --- the three reactions (KEGG + BRENDA verified) ---------------------------
ALANINE_TRANSAMINASE = Reaction(
    key="ALT",
    ec="2.6.1.2",
    recommended_name="alanine transaminase",
    systematic_name="L-alanine:2-oxoglutarate aminotransferase",
    kegg_reaction="R00258",
    amino_donor=L_ALANINE,
    amino_acceptor=OXOGLUTARATE,
    oxo_product=PYRUVATE,
    amino_product=L_GLUTAMATE,
    cofactor=PLP,
)

ASPARTATE_TRANSAMINASE = Reaction(
    key="AST",
    ec="2.6.1.1",
    recommended_name="aspartate transaminase",
    systematic_name="L-aspartate:2-oxoglutarate aminotransferase",
    kegg_reaction="R00355",
    amino_donor=L_ASPARTATE,
    amino_acceptor=OXOGLUTARATE,
    oxo_product=OXALOACETATE,
    amino_product=L_GLUTAMATE,
    cofactor=PLP,
)

CYSTEINE_TRANSAMINASE = Reaction(
    key="CysAT",
    ec="2.6.1.3",
    recommended_name="cysteine transaminase",
    systematic_name="L-cysteine:2-oxoglutarate aminotransferase",
    kegg_reaction="R00895",
    amino_donor=L_CYSTEINE,
    amino_acceptor=OXOGLUTARATE,
    oxo_product=MERCAPTOPYRUVATE,
    amino_product=L_GLUTAMATE,
    cofactor=PLP,
)

REACTIONS: tuple[Reaction, ...] = (
    ALANINE_TRANSAMINASE,
    ASPARTATE_TRANSAMINASE,
    CYSTEINE_TRANSAMINASE,
)

# Namespace for the instances and schema this project mints itself.
BASE_IRI = "https://w3id.org/nfdi4cat/transaminase-kg"
ONTOLOGY_IRI = f"{BASE_IRI}/ontology"


def all_compound_keys() -> set[str]:
    return {c.key for c in COMPOUNDS}


def provenance() -> dict[str, str]:
    """Machine-readable note of where the facts come from (for the graph header)."""
    return {
        "reactions_ec": "KEGG ENZYME; BRENDA",
        "compound_ids": "ChEBI (via KEGG conv/chebi cross-reference)",
        "cofactor": "pyridoxal 5'-phosphate, ChEBI:18405",
    }
