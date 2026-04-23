use async_trait::async_trait;
use std::collections::BTreeMap;

use purpose_core::{Error, Value};

use crate::provider::Provider;

use super::json_to_value;

pub struct UniprotProvider {
    client: reqwest::Client,
    base_url: String,
}

impl UniprotProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://rest.uniprot.org".into(),
        }
    }

    pub fn with_base_url<S: Into<String>>(base_url: S) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
        }
    }
}

impl Default for UniprotProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for UniprotProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        match op {
            "lookup_protein_by_gene" => {
                let gene = args
                    .get("gene")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'gene' argument".into()))?;
                let query = format!(
                    "gene:{} AND organism_id:9606",
                    urlencoding::encode(gene)
                );
                let url = format!(
                    "{}/uniprotkb/search?query={}&format=json&size=1",
                    self.base_url, query
                );
                let resp: serde_json::Value = self
                    .client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| Error::Provider(e.to_string()))?
                    .error_for_status()
                    .map_err(|e| Error::Provider(e.to_string()))?
                    .json()
                    .await
                    .map_err(|e| Error::Provider(e.to_string()))?;

                let first = resp
                    .get("results")
                    .and_then(|r| r.as_array())
                    .and_then(|a| a.first())
                    .ok_or_else(|| {
                        Error::Provider(format!("no UniProt entry for gene {}", gene))
                    })?;

                Ok(json_to_value(first))
            }
            "lookup_protein_by_accession" => {
                let accession = args
                    .get("accession")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'accession' argument".into()))?;
                let url = format!(
                    "{}/uniprotkb/{}?format=json",
                    self.base_url,
                    urlencoding::encode(accession)
                );
                let resp: serde_json::Value = self
                    .client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| Error::Provider(e.to_string()))?
                    .error_for_status()
                    .map_err(|e| Error::Provider(e.to_string()))?
                    .json()
                    .await
                    .map_err(|e| Error::Provider(e.to_string()))?;
                Ok(json_to_value(&resp))
            }
            _ => Err(Error::Provider(format!("unsupported op: {}", op))),
        }
    }
}
