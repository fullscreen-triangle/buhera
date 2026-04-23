use std::collections::HashMap;

use crate::{
    error::Error,
    operation::Operation,
    types::Type,
    vahera::VaHera,
    value::Value,
};

/// Structural type-checker over a vaHera fragment against a registered
/// operation vocabulary. Returns the output `Type` the fragment would
/// produce, or an `Error::Type` describing the first violation.
///
/// The current implementation checks:
///   * Every `Call` names a registered operation.
///   * Every argument key is declared in the operation's input schema.
///   * Literals have types compatible with the expected input (width-typed).
///   * `Compose` pipelines thread the output of step N into step N+1's
///     `input` slot when present.
///   * `Hole` is always rejected; fragments must be fully resolved.
pub fn typecheck(fragment: &VaHera, ops: &HashMap<String, Operation>) -> Result<Type, Error> {
    typecheck_inner(fragment, ops, None)
}

fn typecheck_inner(
    fragment: &VaHera,
    ops: &HashMap<String, Operation>,
    piped: Option<&Type>,
) -> Result<Type, Error> {
    match fragment {
        VaHera::Hole(name) => Err(Error::Type(format!("unresolved hole: {}", name))),
        VaHera::Literal(v) => Ok(value_type(v)),
        VaHera::Call { op, args } => {
            let signature = ops
                .get(op)
                .ok_or_else(|| Error::Type(format!("unknown operation: {}", op)))?;

            for (k, _) in args.iter() {
                if !signature.inputs.contains_key(k) {
                    return Err(Error::Type(format!(
                        "operation {} has no input named {}",
                        op, k
                    )));
                }
            }

            for (name, expected) in signature.inputs.iter() {
                if let Some(child) = args.get(name) {
                    let actual = typecheck_inner(child, ops, None)?;
                    ensure_assignable(&actual, expected).map_err(|msg| {
                        Error::Type(format!(
                            "operation {} argument {}: {}",
                            op, name, msg
                        ))
                    })?;
                } else if name == "input" {
                    match piped {
                        Some(actual) => ensure_assignable(actual, expected).map_err(|msg| {
                            Error::Type(format!(
                                "operation {} piped input: {}",
                                op, msg
                            ))
                        })?,
                        None => {
                            return Err(Error::Type(format!(
                                "operation {} requires input but none was provided",
                                op
                            )));
                        }
                    }
                } else {
                    return Err(Error::Type(format!(
                        "operation {} is missing required argument {}",
                        op, name
                    )));
                }
            }

            Ok(signature.output.clone())
        }
        VaHera::Compose(parts) => {
            let mut carry: Option<Type> = piped.cloned();
            let mut last: Option<Type> = None;
            for p in parts.iter() {
                let out = typecheck_inner(p, ops, carry.as_ref())?;
                carry = Some(out.clone());
                last = Some(out);
            }
            last.ok_or_else(|| Error::Type("empty compose".into()))
        }
    }
}

fn value_type(v: &Value) -> Type {
    match v {
        Value::Null => Type::Unit,
        Value::Bool(_) => Type::Bool,
        Value::Num(_) => Type::Num,
        Value::Str(_) => Type::Str,
        Value::List(_) => Type::List(Box::new(Type::Named("Any".into()))),
        Value::Record(_) => Type::Named("Record".into()),
    }
}

fn ensure_assignable(actual: &Type, expected: &Type) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    match (actual, expected) {
        (_, Type::Var(_)) => Ok(()),
        (Type::Named(_), Type::Named(_)) => Ok(()),
        (Type::List(_), Type::List(inner_expected)) if **inner_expected == Type::Named("Any".into()) => Ok(()),
        _ => Err(format!("expected {:?}, got {:?}", expected, actual)),
    }
}
