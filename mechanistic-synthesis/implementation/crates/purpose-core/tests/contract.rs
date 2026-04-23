use std::collections::{BTreeMap, HashMap};

use purpose_core::{typecheck, Operation, Type, VaHera, Value};

#[test]
fn value_round_trips_through_json() {
    let v = Value::Record(BTreeMap::from([
        ("name".into(), Value::Str("SOD1".into())),
        ("length".into(), Value::Num(154.0)),
    ]));
    let ser = serde_json::to_string(&v).expect("serialize");
    let back: Value = serde_json::from_str(&ser).expect("deserialize");
    assert_eq!(v, back);
}

#[test]
fn vahera_fragment_is_fully_resolved_after_filling_holes() {
    let frag = VaHera::Call {
        op: "lookup".into(),
        args: BTreeMap::from([("name".into(), VaHera::Literal(Value::Str("SOD1".into())))]),
    };
    assert!(frag.is_fully_resolved());
    let with_hole = VaHera::Hole("anything".into());
    assert!(!with_hole.is_fully_resolved());
}

#[test]
fn typecheck_accepts_well_typed_compose() {
    let lookup = Operation::new(
        "lookup_protein_by_gene",
        BTreeMap::from([("gene".into(), Type::Str)]),
        Type::named("ProteinRecord"),
        "",
    );
    let summarize = Operation::new(
        "summarize_protein",
        BTreeMap::from([("input".into(), Type::named("ProteinRecord"))]),
        Type::Str,
        "",
    );
    let mut ops = HashMap::new();
    ops.insert(lookup.name.clone(), lookup);
    ops.insert(summarize.name.clone(), summarize);

    let frag = VaHera::Compose(vec![
        VaHera::Call {
            op: "lookup_protein_by_gene".into(),
            args: BTreeMap::from([(
                "gene".into(),
                VaHera::Literal(Value::Str("SOD1".into())),
            )]),
        },
        VaHera::Call {
            op: "summarize_protein".into(),
            args: BTreeMap::new(),
        },
    ]);
    let out = typecheck(&frag, &ops).expect("typecheck");
    assert_eq!(out, Type::Str);
}

#[test]
fn typecheck_rejects_unknown_operation() {
    let ops: HashMap<String, Operation> = HashMap::new();
    let frag = VaHera::call("no_such_op");
    let err = typecheck(&frag, &ops).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("unknown operation"));
}
