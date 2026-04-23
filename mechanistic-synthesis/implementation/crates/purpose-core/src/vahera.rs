use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::value::Value;

/// The four variants are permanent. New variants may be added in a minor
/// version only if all existing consumers handle them gracefully via a
/// fallback. Variant names are the externally-tagged serde form used on
/// the wire and therefore part of the public contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum VaHera {
    Call {
        op: String,
        #[serde(default)]
        args: BTreeMap<String, VaHera>,
    },
    Compose(Vec<VaHera>),
    Literal(Value),
    Hole(String),
}

impl VaHera {
    pub fn call<N: Into<String>>(name: N) -> Self {
        VaHera::Call {
            op: name.into(),
            args: BTreeMap::new(),
        }
    }

    pub fn literal<V: Into<Value>>(v: V) -> Self {
        VaHera::Literal(v.into())
    }

    pub fn is_fully_resolved(&self) -> bool {
        match self {
            VaHera::Hole(_) => false,
            VaHera::Literal(_) => true,
            VaHera::Call { args, .. } => args.values().all(|a| a.is_fully_resolved()),
            VaHera::Compose(parts) => parts.iter().all(|p| p.is_fully_resolved()),
        }
    }
}

impl From<Value> for VaHera {
    fn from(v: Value) -> Self {
        VaHera::Literal(v)
    }
}
