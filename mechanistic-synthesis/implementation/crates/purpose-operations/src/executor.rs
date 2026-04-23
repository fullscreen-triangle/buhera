use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use purpose_core::{Error, VaHera, Value};

use crate::registry::OperationRegistry;

pub struct Executor {
    registry: OperationRegistry,
}

impl Executor {
    pub fn new(registry: OperationRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &OperationRegistry {
        &self.registry
    }

    pub async fn execute(&self, program: &VaHera) -> Result<Value, Error> {
        self.execute_with(program, None).await
    }

    pub fn execute_with<'a>(
        &'a self,
        program: &'a VaHera,
        piped: Option<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'a>> {
        Box::pin(async move {
            match program {
                VaHera::Hole(name) => Err(Error::Type(format!("unresolved hole: {}", name))),
                VaHera::Literal(v) => Ok(v.clone()),
                VaHera::Call { op, args } => {
                    let (operation, provider) = self
                        .registry
                        .get(op)
                        .ok_or_else(|| Error::Provider(format!("unknown operation: {}", op)))?;

                    let mut resolved: BTreeMap<String, Value> = BTreeMap::new();
                    for (k, frag) in args.iter() {
                        let v = self.execute_with(frag, None).await?;
                        resolved.insert(k.clone(), v);
                    }

                    if operation.inputs.contains_key("input") && !resolved.contains_key("input") {
                        match piped {
                            Some(piped_value) => {
                                resolved.insert("input".into(), piped_value);
                            }
                            None => {
                                return Err(Error::Type(format!(
                                    "operation {} requires piped input but received none",
                                    op
                                )));
                            }
                        }
                    }

                    provider.invoke(op, &resolved).await
                }
                VaHera::Compose(parts) => {
                    let mut carry: Option<Value> = piped;
                    for p in parts.iter() {
                        let out = self.execute_with(p, carry.take()).await?;
                        carry = Some(out);
                    }
                    carry.ok_or_else(|| Error::Internal("empty compose".into()))
                }
            }
        })
    }
}
