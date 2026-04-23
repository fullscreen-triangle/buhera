use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("compile error: {0}")]
    Compile(String),
    #[error("type error: {0}")]
    Type(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("internal error: {0}")]
    Internal(String),
}
