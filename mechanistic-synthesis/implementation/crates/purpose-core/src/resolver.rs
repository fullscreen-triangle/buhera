use async_trait::async_trait;

use crate::{error::Error, vahera::VaHera};

/// Utterance -> vaHera fragment. Permanent signature.
#[async_trait]
pub trait Resolver: Send + Sync {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error>;
}
