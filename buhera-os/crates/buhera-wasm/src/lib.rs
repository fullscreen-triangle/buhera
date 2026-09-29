//! `buhera-wasm` — the pure Rust modules behind a C ABI, for the TypeScript
//! host (specification `07-bindings.md` §4, "native via wasm").
//!
//! Every call takes one UTF-8 JSON document in linear memory and returns one
//! UTF-8 JSON document, packed as `(ptr << 32) | len`. The caller frees the
//! returned buffer with `bw_free`. The JS side is
//! `buhera-os/registry-ts/src/wasm.ts`.
//!
//! | export | input | output |
//! |---|---|---|
//! | `bw_describe` | — | `{ modules: Descriptor[], dsls: DslSummary[] }` |
//! | `bw_dispatch` | `{ module, instruction, act_budget }` | `ActResult`, or `{ error }` for an unknown module |
//! | `bw_validate` | `{ dsl, source }` | `Validation`, or `{ error }` for an unknown language |
//!
//! Acts are executed through `Registry::execute_unaudited`: the *calling*
//! registry (TypeScript) is the one that audits, so an act is recorded
//! exactly once — and no clock is read, which matters because
//! `std::time::Instant::now` panics on `wasm32-unknown-unknown`.
//!
//! `unsafe` is confined to the two allocation shims and pointer→slice
//! conversion at the boundary; everything past it is safe Rust.

use std::cell::RefCell;

use buhera_registry::{DslRegistry, Registry};
use serde_json::{json, Value};

thread_local! {
    static FED: RefCell<(Registry, DslRegistry)> = RefCell::new(wasm_federation());
}

/// The wasm-safe modules, registered explicitly rather than through
/// `buhera_modules::federation`: Cargo unifies features across a workspace
/// build, so relying on this crate's feature list would let non-wasm-safe
/// modules (vahera's kernel, sbs-core's clock) in under `cargo test --workspace`.
fn wasm_federation() -> (Registry, DslRegistry) {
    use buhera_modules::{heihachi, levinthal, ndombolo, olduvai, tracker, windtunnel};
    let mut modules = Registry::new();
    let mut dsls = DslRegistry::new();
    modules.register(Box::new(ndombolo::Ndombolo::new()));
    dsls.register(ndombolo::dsl());
    modules.register(Box::new(windtunnel::WindTunnel::new()));
    dsls.register(windtunnel::dsl());
    modules.register(Box::new(tracker::Tracker { filesystem: false }));
    modules.register(Box::new(heihachi::Heihachi::new()));
    for d in heihachi::dsls() {
        dsls.register(d);
    }
    modules.register(Box::new(olduvai::Olduvai::new()));
    modules.register(Box::new(levinthal::Levinthal::new()));
    (modules, dsls)
}

/// Allocate `len` bytes for the host to write an input document into.
#[no_mangle]
pub extern "C" fn bw_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Free a buffer previously returned by `bw_alloc` or by a call.
///
/// # Safety
/// `ptr`/`len` must come from `bw_alloc(len)` or a packed call result.
#[no_mangle]
pub unsafe extern "C" fn bw_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
    }
}

fn read(ptr: *const u8, len: usize) -> Value {
    // SAFETY: the host wrote `len` bytes at `ptr` (from `bw_alloc`).
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    serde_json::from_slice(bytes).unwrap_or(Value::Null)
}

fn write(v: &Value) -> u64 {
    let mut bytes = serde_json::to_vec(v).unwrap_or_else(|_| b"null".to_vec());
    bytes.shrink_to_fit();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ((ptr as u64) << 32) | len as u64
}

/// Pure implementation of `bw_describe`, testable natively.
pub fn describe() -> Value {
    FED.with(|f| {
        let f = f.borrow();
        json!({ "modules": f.0.list(), "dsls": f.1.list() })
    })
}

/// Pure implementation of `bw_dispatch`.
pub fn dispatch(input: &Value) -> Value {
    let module = input.get("module").and_then(Value::as_str).unwrap_or("");
    let instruction = input.get("instruction").cloned().unwrap_or(Value::Null);
    let budget = input.get("act_budget").and_then(Value::as_u64).unwrap_or(1).max(1) as u32;
    FED.with(|f| {
        match f.borrow_mut().0.execute_unaudited(module, &instruction, budget) {
            Ok(r) => serde_json::to_value(r).unwrap_or(Value::Null),
            Err(e) => json!({ "error": e.to_string() }),
        }
    })
}

/// Pure implementation of `bw_validate`.
pub fn validate(input: &Value) -> Value {
    let dsl = input.get("dsl").and_then(Value::as_str).unwrap_or("");
    let source = input.get("source").and_then(Value::as_str).unwrap_or("");
    FED.with(|f| match f.borrow().1.validate(dsl, source) {
        Ok(v) => serde_json::to_value(v).unwrap_or(Value::Null),
        Err(e) => json!({ "error": e.to_string() }),
    })
}

/// C ABI: describe the modules and languages in this build.
#[no_mangle]
pub extern "C" fn bw_describe() -> u64 {
    write(&describe())
}

/// C ABI: dispatch one act.
#[no_mangle]
pub extern "C" fn bw_dispatch(ptr: *const u8, len: usize) -> u64 {
    write(&dispatch(&read(ptr, len)))
}

/// C ABI: validate source against a registered language.
#[no_mangle]
pub extern "C" fn bw_validate(ptr: *const u8, len: usize) -> u64 {
    write(&validate(&read(ptr, len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_exactly_the_pure_modules() {
        let d = describe();
        let ids: Vec<&str> = d["modules"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["heihachi", "levinthal", "ndombolo", "olduvai", "tracker", "windtunnel"]);
    }

    #[test]
    fn dispatch_round_trips_and_does_not_retain_audit() {
        let r = dispatch(&json!({ "module": "ndombolo", "instruction": "item x = 2\n// ---\nprint(x * 3)" }));
        assert_eq!(r["ok"], true);
        assert_eq!(r["output_delta"]["cells"][1]["output"][0], "6");
        FED.with(|f| assert!(f.borrow().0.audit_log().is_empty()));
    }

    #[test]
    fn unknown_module_and_language_are_errors() {
        assert!(dispatch(&json!({ "module": "nope" }))["error"].is_string());
        assert!(validate(&json!({ "dsl": "nope", "source": "" }))["error"].is_string());
    }
}
