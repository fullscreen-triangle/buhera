use async_trait::async_trait;
use std::collections::BTreeMap;

use purpose_core::{Error, Value};

use crate::provider::Provider;

/// Purely functional transformation: takes a UniProt record, produces a
/// human-readable summary string. No I/O.
pub struct ProteinSummaryProvider;

#[async_trait]
impl Provider for ProteinSummaryProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        if op != "summarize_protein" {
            return Err(Error::Provider(format!("unsupported op: {}", op)));
        }
        let record = args
            .get("input")
            .ok_or_else(|| Error::Provider("missing 'input' argument".into()))?;
        let rec = record.as_record().ok_or_else(|| {
            Error::Provider("expected Record input for summarize_protein".into())
        })?;

        let accession = rec
            .get("primaryAccession")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let name = rec
            .get("proteinDescription")
            .and_then(|v| v.as_record())
            .and_then(|d| d.get("recommendedName"))
            .and_then(|v| v.as_record())
            .and_then(|n| n.get("fullName"))
            .and_then(|v| v.as_record())
            .and_then(|f| f.get("value"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown protein");
        let length = rec
            .get("sequence")
            .and_then(|v| v.as_record())
            .and_then(|s| s.get("length"))
            .and_then(|v| v.as_num())
            .map(|n| n as u64)
            .unwrap_or(0);
        let organism = rec
            .get("organism")
            .and_then(|v| v.as_record())
            .and_then(|o| o.get("scientificName"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown organism");

        let summary = format!(
            "{} ({}) — {} residues, {}.",
            name, accession, length, organism
        );
        Ok(Value::Str(summary))
    }
}
