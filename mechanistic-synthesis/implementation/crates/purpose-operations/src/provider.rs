use async_trait::async_trait;
use std::collections::BTreeMap;

use purpose_core::{Error, Value};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error>;
}
