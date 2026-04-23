use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Type {
    Str,
    Num,
    Bool,
    Unit,
    List(Box<Type>),
    Named(String),
    Var(String),
}

impl Type {
    pub fn named<S: Into<String>>(s: S) -> Self {
        Type::Named(s.into())
    }

    pub fn var<S: Into<String>>(s: S) -> Self {
        Type::Var(s.into())
    }

    pub fn list(inner: Type) -> Self {
        Type::List(Box::new(inner))
    }
}
