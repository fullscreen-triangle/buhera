//! Phase-1 acceptance test: with all-noop subsystems, the kernel must produce
//! results bit-for-bit identical to the bare `Executor`.
//!
//! Uses a hermetic local provider (no network) so the test is deterministic
//! and fast.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use purpose_core::{Error, Operation, Type, VaHera, Value};
use purpose_kernel::BuheraKernel;
use purpose_operations::{Executor, OperationRegistry, Provider};

/// Local test provider with three deterministic ops:
/// * `make_record(name)` → `Record { name, count: 0 }`
/// * `bump(input)` → input record with `count` incremented
/// * `summarize(input)` → `Str("name=<n>; count=<c>")`
///
/// Together they form a typical Compose chain that exercises the executor's
/// pipe-threading.
struct LocalProvider;

#[async_trait]
impl Provider for LocalProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        match op {
            "make_record" => {
                let name = args
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Type("make_record: missing 'name'".into()))?
                    .to_string();
                let mut rec = BTreeMap::new();
                rec.insert("name".into(), Value::Str(name));
                rec.insert("count".into(), Value::Num(0.0));
                Ok(Value::Record(rec))
            }
            "bump" => {
                let rec = args
                    .get("input")
                    .and_then(Value::as_record)
                    .ok_or_else(|| Error::Type("bump: missing 'input' record".into()))?
                    .clone();
                let mut next = rec.clone();
                let current = next
                    .get("count")
                    .and_then(Value::as_num)
                    .unwrap_or(0.0);
                next.insert("count".into(), Value::Num(current + 1.0));
                Ok(Value::Record(next))
            }
            "summarize" => {
                let rec = args
                    .get("input")
                    .and_then(Value::as_record)
                    .ok_or_else(|| Error::Type("summarize: missing 'input' record".into()))?;
                let name = rec
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
                let count = rec
                    .get("count")
                    .and_then(Value::as_num)
                    .unwrap_or(0.0);
                Ok(Value::Str(format!("name={}; count={}", name, count)))
            }
            _ => Err(Error::Provider(format!("unknown op: {}", op))),
        }
    }
}

fn build_registry() -> OperationRegistry {
    let mut registry = OperationRegistry::new();
    let provider: Arc<dyn Provider> = Arc::new(LocalProvider);

    let mut make_inputs = BTreeMap::new();
    make_inputs.insert("name".into(), Type::Str);
    registry.register(
        Operation::new(
            "make_record",
            make_inputs,
            Type::named("CounterRecord"),
            "Construct a fresh counter record with the given name.",
        ),
        provider.clone(),
    );

    let mut bump_inputs = BTreeMap::new();
    bump_inputs.insert("input".into(), Type::named("CounterRecord"));
    registry.register(
        Operation::new(
            "bump",
            bump_inputs,
            Type::named("CounterRecord"),
            "Increment the counter on a record.",
        ),
        provider.clone(),
    );

    let mut summary_inputs = BTreeMap::new();
    summary_inputs.insert("input".into(), Type::named("CounterRecord"));
    registry.register(
        Operation::new(
            "summarize",
            summary_inputs,
            Type::Str,
            "Render the record as a string.",
        ),
        provider,
    );

    registry
}

/// Build a Compose fragment: make_record("alpha") | bump | bump | summarize.
fn fragment() -> VaHera {
    let mut make_args: BTreeMap<String, VaHera> = BTreeMap::new();
    make_args.insert("name".into(), VaHera::Literal(Value::Str("alpha".into())));
    VaHera::Compose(vec![
        VaHera::Call {
            op: "make_record".into(),
            args: make_args,
        },
        VaHera::Call {
            op: "bump".into(),
            args: BTreeMap::new(),
        },
        VaHera::Call {
            op: "bump".into(),
            args: BTreeMap::new(),
        },
        VaHera::Call {
            op: "summarize".into(),
            args: BTreeMap::new(),
        },
    ])
}

#[tokio::test]
async fn kernel_matches_bare_executor() {
    let frag = fragment();

    let bare = Executor::new(build_registry());
    let bare_value = bare.execute(&frag).await.expect("bare executor failed");

    let kernel = BuheraKernel::builder(build_registry()).build();
    let kernel_value = kernel.dispatch(&frag).await.expect("kernel dispatch failed");

    assert_eq!(bare_value, kernel_value);
    assert_eq!(
        kernel_value.as_str(),
        Some("name=alpha; count=2"),
        "expected pipeline result"
    );
}

#[tokio::test]
async fn kernel_publishes_dispatch_event() {
    let kernel = BuheraKernel::builder(build_registry()).build();
    let mut rx = kernel.events().subscribe();

    let value = kernel.dispatch(&fragment()).await.expect("dispatch failed");
    assert_eq!(value.as_str(), Some("name=alpha; count=2"));

    let event = rx
        .try_recv()
        .expect("expected exactly one dispatch event after one dispatch");
    assert_eq!(event.op.as_deref(), Some("summarize"));
    assert!(event.ok);
    assert!(!event.cache_hit, "Phase 1 CMM never hits");
    assert_eq!(event.decisions_consumed, 1);
}

#[tokio::test]
async fn kernel_propagates_provider_errors() {
    let kernel = BuheraKernel::builder(build_registry()).build();

    let mut bad_args: BTreeMap<String, VaHera> = BTreeMap::new();
    bad_args.insert("name".into(), VaHera::Literal(Value::Num(42.0)));
    let bad = VaHera::Call {
        op: "make_record".into(),
        args: bad_args,
    };

    let result = kernel.dispatch(&bad).await;
    assert!(matches!(result, Err(Error::Type(_))));
}

#[tokio::test]
async fn kernel_dispatches_literal_unchanged() {
    let kernel = BuheraKernel::builder(build_registry()).build();
    let lit = VaHera::Literal(Value::Num(7.0));
    let value = kernel.dispatch(&lit).await.expect("literal dispatch failed");
    assert_eq!(value.as_num(), Some(7.0));
}
