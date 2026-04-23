use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Runtime value threaded through the executor. The six variants are frozen.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Record(BTreeMap<String, Value>),
}

impl Value {
    pub fn str<S: Into<String>>(s: S) -> Self {
        Value::Str(s.into())
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    pub fn as_num(&self) -> Option<f64> {
        if let Value::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        if let Value::List(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    pub fn as_record(&self) -> Option<&BTreeMap<String, Value>> {
        if let Value::Record(m) = self {
            Some(m)
        } else {
            None
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}
