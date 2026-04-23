use std::sync::Arc;

use crate::{operation::Operation, resolver::Resolver};

#[derive(Clone)]
pub struct Domain {
    pub name: String,
    pub operations: Vec<Operation>,
    pub resolver: Arc<dyn Resolver>,
}
