//! Reference providers shipped with the MVP.
//!
//! Domain crates are free to register their own providers; these are
//! reused by `purpose-domains-protein` as the default wiring.

mod uniprot;
mod summary;

pub use uniprot::UniprotProvider;
pub use summary::ProteinSummaryProvider;

use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

use purpose_core::Value;

pub(crate) fn json_to_value(j: &JsonValue) -> Value {
    match j {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(b) => Value::Bool(*b),
        JsonValue::Number(n) => Value::Num(n.as_f64().unwrap_or(0.0)),
        JsonValue::String(s) => Value::Str(s.clone()),
        JsonValue::Array(a) => Value::List(a.iter().map(json_to_value).collect()),
        JsonValue::Object(m) => {
            let mut out: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in m.iter() {
                out.insert(k.clone(), json_to_value(v));
            }
            Value::Record(out)
        }
    }
}
