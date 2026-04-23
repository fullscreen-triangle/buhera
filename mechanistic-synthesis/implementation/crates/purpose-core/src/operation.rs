use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::types::Type;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub inputs: BTreeMap<String, Type>,
    pub output: Type,
    pub description: String,
}

impl Operation {
    pub fn new<N, D>(name: N, inputs: BTreeMap<String, Type>, output: Type, description: D) -> Self
    where
        N: Into<String>,
        D: Into<String>,
    {
        Operation {
            name: name.into(),
            inputs,
            output,
            description: description.into(),
        }
    }
}
