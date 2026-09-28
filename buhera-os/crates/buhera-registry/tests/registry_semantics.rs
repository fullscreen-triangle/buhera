//! Registry semantics R1–R6 (specification `03-registry.md` §4). The
//! TypeScript library runs the same six cases in
//! `registry-ts/test/registry-semantics.test.ts`.

use std::sync::{Arc, Mutex};

use buhera_registry::{
    ActResult, BindingKind, Descriptor, DispatchError, DslEntry, DslError, DslRegistry, Instruction, Module,
    Registry, Validation,
};
use serde_json::json;

struct Echo {
    calls: u32,
}

impl Module for Echo {
    fn id(&self) -> &str {
        "echo"
    }
    fn describe(&self) -> Descriptor {
        Descriptor {
            id: "echo".into(),
            description: "echo".into(),
            instructions: vec![],
            dsl: None,
            binding: BindingKind::Native,
        }
    }
    fn execute(&mut self, instruction: &Instruction, _budget: u32) -> ActResult {
        self.calls += 1;
        ActResult::done(json!({ "kind": "echo", "value": instruction, "calls": self.calls }), 0.0)
    }
}

struct Panicky;

impl Module for Panicky {
    fn id(&self) -> &str {
        "panicky"
    }
    fn describe(&self) -> Descriptor {
        Descriptor {
            id: "panicky".into(),
            description: "always panics".into(),
            instructions: vec![],
            dsl: None,
            binding: BindingKind::Native,
        }
    }
    fn execute(&mut self, _i: &Instruction, _b: u32) -> ActResult {
        panic!("boom")
    }
}

#[test]
fn r1_unknown_module_is_a_caller_error_and_is_not_audited() {
    let mut r = Registry::new();
    let err = r.dispatch("nope", json!("x"), 1).unwrap_err();
    assert_eq!(err, DispatchError::UnknownModule("nope".into()));
    assert!(r.audit_log().is_empty());
}

#[test]
fn r2_panics_are_contained_with_null_delta() {
    let mut r = Registry::new();
    r.register(Box::new(Panicky));
    let res = r.dispatch("panicky", json!({}), 1).unwrap();
    assert!(!res.ok);
    assert!(res.output_delta.is_none());
    assert_eq!(res.residue, 0.0);
    assert!(res.completed);
    assert_eq!(res.error.as_deref(), Some("boom"));
    assert_eq!(r.audit_log().len(), 1);
}

#[test]
fn r3_act_ids_are_monotone_and_survive_clearing() {
    let mut r = Registry::new();
    r.register(Box::new(Echo { calls: 0 }));
    r.register(Box::new(Panicky));
    r.dispatch("echo", json!(1), 1).unwrap();
    r.dispatch("panicky", json!(2), 1).unwrap();
    r.clear_audit_log();
    r.dispatch("echo", json!(3), 1).unwrap();
    assert_eq!(r.audit_log()[0].act_id, 3);
}

#[test]
fn r4_hooks_run_in_order_and_a_failing_hook_is_isolated() {
    let mut r = Registry::new();
    r.register(Box::new(Echo { calls: 0 }));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let a = Arc::clone(&seen);
    r.on_dispatch(move |e| a.lock().unwrap().push(format!("a{}", e.act_id)));
    r.on_dispatch(|_| panic!("hook failure"));
    let b = Arc::clone(&seen);
    let hb = r.on_dispatch(move |e| b.lock().unwrap().push(format!("b{}", e.act_id)));
    assert!(r.dispatch("echo", json!("x"), 1).unwrap().ok);
    assert!(r.remove_hook(hb));
    r.dispatch("echo", json!("y"), 1).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec!["a1", "b1", "a2"]);
}

#[test]
fn r5_register_replaces_and_returns_the_previous_binding() {
    let mut r = Registry::new();
    assert!(r.register(Box::new(Echo { calls: 0 })).is_none());
    r.dispatch("echo", json!(1), 1).unwrap();
    let prev = r.register(Box::new(Echo { calls: 100 }));
    assert!(prev.is_some());
    let res = r.dispatch("echo", json!(1), 1).unwrap();
    assert_eq!(res.output_delta.unwrap()["calls"], 101);
}

#[test]
fn r6_module_state_persists_between_acts() {
    let mut r = Registry::new();
    r.register(Box::new(Echo { calls: 0 }));
    for _ in 0..3 {
        r.dispatch("echo", json!(null), 1).unwrap();
    }
    let e = &r.audit_log()[2];
    assert_eq!(e.result.output_delta.as_ref().unwrap()["calls"], 3);
    assert_eq!(e.act_budget, 1);
    assert!(e.timestamp.ends_with('Z'));
}

fn validate_balanced(src: &str) -> Validation {
    let mut depth = 0i32;
    for (i, line) in src.lines().enumerate() {
        for c in line.chars() {
            depth += match c {
                '{' => 1,
                '}' => -1,
                _ => 0,
            };
            if depth < 0 {
                return Validation::invalid(vec![DslError::at("unbalanced }", i as u32 + 1, None)]);
            }
        }
    }
    if depth == 0 {
        Validation::valid()
    } else {
        Validation::invalid(vec![DslError::msg("unclosed {")])
    }
}

#[test]
fn d1_dsl_registry_routes_and_validates() {
    let mut d = DslRegistry::new();
    d.register(DslEntry {
        id: "braces",
        label: "Braces",
        extension: ".br",
        module_id: "echo",
        pack_id: "braces",
        validate: validate_balanced,
    });
    assert!(d.validate("braces", "a { b }").unwrap().ok);
    let bad = d.validate("braces", "a\n}").unwrap();
    assert_eq!(bad.errors[0].line, Some(2));
    assert!(d.validate("nope", "").is_err());
    assert_eq!(d.by_extension(".BR").unwrap().module_id, "echo");
    assert_eq!(buhera_registry::line_from_message("line 12: unexpected token"), Some(12));
}
