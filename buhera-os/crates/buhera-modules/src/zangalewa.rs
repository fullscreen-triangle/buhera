//! `zangalewa` — the OS's only AI module: `(dslId, instructions)` → DSL
//! chunks that the owning module's own compiler accepts (specification
//! `specs/zangalewa.md`). Wraps vendored `zangalewa-dsl`; reimplements
//! nothing. Every accepted draft is returned — the module never picks a
//! winner and makes no semantic judgement.
//!
//! Side effects: network calls to Ollama / OpenAI / Anthropic / Gemini
//! (whichever are configured through upstream's environment variables),
//! and a one-time read of knowledge packs (`ZANGALEWA_PACKS`). Output is
//! nondeterministic (sampled at T ≥ 0.2); only validation is deterministic.

use buhera_registry::{field_str, instruction_kind, ActResult, BindingKind, Descriptor, Instruction, Module};
use serde_json::{json, Value};
use zangalewa_dsl::{generate, get_dsl, list_dsls, provider_status, GenerateRequest, Stage};

/// Registry id.
pub const ID: &str = "zangalewa-dsl";

/// The module. Owns the tokio runtime `generate` needs (contract M5).
pub struct Zangalewa {
    rt: Option<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for Zangalewa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Zangalewa").field("runtime", &self.rt.is_some()).finish()
    }
}

impl Default for Zangalewa {
    fn default() -> Self {
        Self::new()
    }
}

impl Zangalewa {
    /// New module; the runtime is created lazily on the first `generate`.
    pub fn new() -> Self {
        Self { rt: None }
    }

    fn runtime(&mut self) -> Result<&tokio::runtime::Runtime, String> {
        if self.rt.is_none() {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("cannot start runtime: {e}"))?;
            self.rt = Some(rt);
        }
        Ok(self.rt.as_ref().expect("just set"))
    }
}

impl Module for Zangalewa {
    fn id(&self) -> &str {
        ID
    }

    fn describe(&self) -> Descriptor {
        Descriptor {
            id: ID.into(),
            description: "Zangalewa — natural-language instructions → DSL chunks that the owning module's own \
                          compiler accepts (generate → validate → repair, every accepted draft returned). \
                          Calls configured model providers."
                .into(),
            instructions: vec![
                r#"dispatch("zangalewa-dsl", { kind: "generate", dslId: "vahera", instructions: "store a note about Friday" })"#
                    .into(),
                r#"dispatch("zangalewa-dsl", "providers")"#.into(),
                r#"dispatch("zangalewa-dsl", { kind: "validate", dslId: "vahera", source: "memory list" })"#.into(),
            ],
            dsl: None,
            binding: BindingKind::Native,
        }
    }

    fn execute(&mut self, instruction: &Instruction, act_budget: u32) -> ActResult {
        match instruction_kind(instruction) {
            Some("providers") => {
                let rows: Vec<Value> = provider_status()
                    .into_iter()
                    .map(|(id, label, available, cost)| json!({ "id": id, "label": label, "available": available, "cost": cost }))
                    .collect();
                let dsls: Vec<Value> = list_dsls()
                    .iter()
                    .map(|d| json!({ "id": d.id, "label": d.label, "module_id": d.module_id, "pack_id": d.pack_id, "accepts_fragment": d.accepts_fragment }))
                    .collect();
                ActResult::done(
                    json!({ "kind": "zangalewa_providers", "summary": format!("zangalewa: {} provider(s), {} DSL(s)", rows.len(), dsls.len()),
                            "providers": rows, "dsls": dsls }),
                    0.0,
                )
            }
            Some("validate") => {
                let (Some(dsl_id), Some(source)) = (field_str(instruction, "dslId"), field_str(instruction, "source")) else {
                    return ActResult::invalid(ID, "{ kind: \"validate\", dslId, source }");
                };
                match get_dsl(dsl_id) {
                    Some(d) => {
                        let v = (d.validate)(source);
                        let n = v.errors.len();
                        ActResult::done(json!({ "kind": "dsl_validation", "dsl": dsl_id, "ok": v.ok, "errors": v.errors }), n as f64)
                    }
                    None => ActResult::fail(&[&format!("zangalewa: unknown dsl: {dsl_id}")], "unknown dsl"),
                }
            }
            Some("generate") => {
                let mut req: GenerateRequest = match serde_json::from_value(instruction.clone()) {
                    Ok(r) => r,
                    Err(e) => return ActResult::fail(&[&format!("zangalewa: bad generate request: {e}")], "invalid instruction"),
                };
                // One act-budget unit = one draft (contract M6); never more
                // drafts than the caller asked for.
                req.drafts = Some(req.drafts.unwrap_or(1).min(act_budget.max(1)));
                let rt = match self.runtime() {
                    Ok(rt) => rt,
                    Err(e) => return ActResult::fail(&[&format!("zangalewa: {e}")], "runtime unavailable"),
                };
                let result = rt.block_on(generate(req));
                let accepted = result.chunks.len();
                let rejected = result.rejected.len();
                let retryable = matches!(result.stage, Some(Stage::Provider));
                let mut delta = serde_json::to_value(&result).unwrap_or(Value::Null);
                if let Value::Object(m) = &mut delta {
                    m.insert("kind".into(), json!("zangalewa_generated"));
                    m.insert(
                        "summary".into(),
                        json!(format!("zangalewa: {accepted} chunk(s) compiled, {rejected} rejected")),
                    );
                    m.insert("retryable".into(), json!(retryable));
                }
                // Residue is syntactic only: the fraction of drafts the
                // owning compiler refused. It says nothing about meaning.
                let residue = if accepted + rejected == 0 { 1.0 } else { rejected as f64 / (accepted + rejected) as f64 };
                ActResult {
                    ok: result.ok,
                    output_delta: Some(delta),
                    residue,
                    completed: true,
                    error: if result.ok { None } else { result.error.clone().or(Some("no draft compiled".into())) },
                }
            }
            _ => ActResult::invalid(ID, "{ kind: generate|validate|providers, … }"),
        }
    }
}
