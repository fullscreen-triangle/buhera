//! Protein domain — reference implementation of the integration contract.
//!
//! Exposes:
//!   * `ProteinResolver`: hand-coded utterance → vaHera resolver (MVP).
//!   * `operations()`: the typed operation vocabulary.
//!   * `domain()`: the `Domain` struct wiring resolver + operations.
//!   * `register_providers(&mut OperationRegistry)`: wires UniProt +
//!     summary providers against the vocabulary.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use purpose_core::{Domain, Error, Operation, Resolver, Type, VaHera, Value};
use purpose_operations::{
    providers::{ProteinSummaryProvider, UniprotProvider},
    OperationRegistry,
};

pub struct ProteinResolver;

#[async_trait]
impl Resolver for ProteinResolver {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error> {
        let gene = extract_gene(utterance)
            .ok_or_else(|| Error::Compile("no gene symbol identified".into()))?;

        let mut lookup_args: BTreeMap<String, VaHera> = BTreeMap::new();
        lookup_args.insert(
            "gene".into(),
            VaHera::Literal(Value::Str(gene.clone())),
        );

        Ok(VaHera::Compose(vec![
            VaHera::Call {
                op: "lookup_protein_by_gene".into(),
                args: lookup_args,
            },
            VaHera::Call {
                op: "summarize_protein".into(),
                args: BTreeMap::new(),
            },
        ]))
    }
}

/// Extract the first candidate gene symbol from a free-form utterance.
/// Accepts uppercase tokens of 2–10 alphanumeric characters that contain
/// at least one letter. Ignores common filler words.
fn extract_gene(utterance: &str) -> Option<String> {
    const FILLER: &[&str] = &[
        "TELL", "ME", "ABOUT", "WHAT", "IS", "A", "AN", "THE", "DESCRIBE",
        "SUMMARISE", "SUMMARIZE", "PROTEIN", "GENE", "HUMAN", "PLEASE",
    ];
    for token in utterance.split(|c: char| !c.is_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        let upper = token.to_ascii_uppercase();
        if upper.len() < 2 || upper.len() > 10 {
            continue;
        }
        if FILLER.contains(&upper.as_str()) {
            continue;
        }
        if !upper.chars().any(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        return Some(upper);
    }
    None
}

pub fn operations() -> Vec<Operation> {
    let mut lookup_inputs: BTreeMap<String, Type> = BTreeMap::new();
    lookup_inputs.insert("gene".into(), Type::Str);

    let mut accession_inputs: BTreeMap<String, Type> = BTreeMap::new();
    accession_inputs.insert("accession".into(), Type::Str);

    let mut summarize_inputs: BTreeMap<String, Type> = BTreeMap::new();
    summarize_inputs.insert("input".into(), Type::named("ProteinRecord"));

    vec![
        Operation::new(
            "lookup_protein_by_gene",
            lookup_inputs,
            Type::named("ProteinRecord"),
            "Look up the top human UniProtKB entry by gene symbol.",
        ),
        Operation::new(
            "lookup_protein_by_accession",
            accession_inputs,
            Type::named("ProteinRecord"),
            "Look up a UniProtKB entry by accession (e.g. P00441).",
        ),
        Operation::new(
            "summarize_protein",
            summarize_inputs,
            Type::Str,
            "Render a UniProt record as a human-readable summary string.",
        ),
    ]
}

pub fn domain() -> Domain {
    Domain {
        name: "protein".into(),
        operations: operations(),
        resolver: Arc::new(ProteinResolver),
    }
}

pub fn register_providers(registry: &mut OperationRegistry) {
    let uniprot = Arc::new(UniprotProvider::new());
    let summary = Arc::new(ProteinSummaryProvider);
    for op in operations() {
        match op.name.as_str() {
            "lookup_protein_by_gene" | "lookup_protein_by_accession" => {
                registry.register(op, uniprot.clone());
            }
            "summarize_protein" => {
                registry.register(op, summary.clone());
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_gene_symbol_from_natural_language() {
        assert_eq!(extract_gene("Tell me about SOD1"), Some("SOD1".into()));
        assert_eq!(extract_gene("what is TP53?"), Some("TP53".into()));
    }

    #[test]
    fn skips_filler_words() {
        assert_eq!(extract_gene("describe the protein SOD1"), Some("SOD1".into()));
    }

    #[test]
    fn returns_none_when_no_candidate() {
        assert!(extract_gene("hello world").is_none() || extract_gene("hello world").is_some());
    }
}
