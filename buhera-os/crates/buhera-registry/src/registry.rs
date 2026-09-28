//! The module registry (specification `03-registry.md`).
//!
//! One entry point, [`Registry::dispatch`]. Every act it performs is
//! appended to the audit log, then handed to every post-dispatch hook.
//! Semantics are identical to `@buhera/registry`'s `Registry`:
//!
//! 1. Unknown module → [`DispatchError::UnknownModule`] (the TS side throws).
//!    This is a caller error, distinct from a module that fails.
//! 2. A module that panics is contained: the act is recorded as
//!    `{ok: false, output_delta: null, residue: 0, completed: true, error}`.
//! 3. `act_id` starts at 1 and increases by one per dispatch, never reused,
//!    including across failed acts.
//! 4. Hooks run after the audit entry is appended, in registration order.
//!    A panicking hook is contained and does not affect other hooks or the
//!    caller.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::contract::{ActResult, Descriptor, Instruction, Module};

/// One dispatched act, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotone act number, from 1.
    pub act_id: u64,
    /// The module the act was dispatched to.
    pub module_id: String,
    /// The instruction, verbatim.
    pub instruction: Instruction,
    /// The act budget the caller granted.
    pub act_budget: u32,
    /// What the module returned (or the contained-panic result).
    pub result: ActResult,
    /// Wall-clock duration of `execute`.
    pub wall_clock_ms: u64,
    /// RFC 3339 UTC timestamp of completion.
    pub timestamp: String,
}

/// Why a dispatch could not be performed at all.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DispatchError {
    /// No module is registered under this id.
    #[error("dispatch: unknown module \"{0}\"")]
    UnknownModule(String),
}

/// Handle returned by [`Registry::on_dispatch`]; pass to
/// [`Registry::remove_hook`] to unregister.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HookId(u64);

type Hook = Box<dyn FnMut(&AuditEntry) + Send>;

/// The federation.
pub struct Registry {
    modules: BTreeMap<String, Box<dyn Module>>,
    audit: Vec<AuditEntry>,
    hooks: Vec<(HookId, Hook)>,
    next_act: u64,
    next_hook: u64,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self { modules: BTreeMap::new(), audit: Vec::new(), hooks: Vec::new(), next_act: 1, next_hook: 1 }
    }

    /// Bind a module under its id. Returns the module previously bound to
    /// the same id, if any (the binding is replaced, as in TS).
    pub fn register(&mut self, module: Box<dyn Module>) -> Option<Box<dyn Module>> {
        let id = module.id().to_string();
        self.modules.insert(id, module)
    }

    /// Remove a module binding.
    pub fn unregister(&mut self, module_id: &str) -> Option<Box<dyn Module>> {
        self.modules.remove(module_id)
    }

    /// Registered module ids, sorted.
    pub fn ids(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// Every module's self-description, sorted by id.
    pub fn list(&self) -> Vec<Descriptor> {
        self.modules.values().map(|m| m.describe()).collect()
    }

    /// Is a module registered under `module_id`?
    pub fn contains(&self, module_id: &str) -> bool {
        self.modules.contains_key(module_id)
    }

    /// Dispatch one act. See the module docs for the exact semantics.
    pub fn dispatch(
        &mut self,
        module_id: &str,
        instruction: Instruction,
        act_budget: u32,
    ) -> Result<ActResult, DispatchError> {
        let module = self
            .modules
            .get_mut(module_id)
            .ok_or_else(|| DispatchError::UnknownModule(module_id.to_string()))?;

        let t0 = Instant::now();
        let result = match catch_unwind(AssertUnwindSafe(|| module.execute(&instruction, act_budget))) {
            Ok(r) => r,
            Err(payload) => ActResult {
                ok: false,
                output_delta: None,
                residue: 0.0,
                completed: true,
                error: Some(panic_message(&payload)),
            },
        };

        let entry = AuditEntry {
            act_id: self.next_act,
            module_id: module_id.to_string(),
            instruction,
            act_budget,
            result: result.clone(),
            wall_clock_ms: t0.elapsed().as_millis() as u64,
            timestamp: rfc3339_now(),
        };
        self.next_act += 1;
        self.audit.push(entry);
        let entry = self.audit.last().expect("just pushed");

        for (_, hook) in self.hooks.iter_mut() {
            // Best-effort: a failing hook never breaks the dispatch.
            let _ = catch_unwind(AssertUnwindSafe(|| hook(entry)));
        }
        Ok(result)
    }

    /// Execute one act **without** auditing, timing, or hooks — for a
    /// registry that serves as the engine room of another registry which does
    /// the auditing (the wasm build under `@buhera/registry`, a remote host's
    /// forwarded act). Panics are still contained (R2). Reads no clock, so it
    /// is safe on `wasm32-unknown-unknown`, where `Instant::now` panics.
    pub fn execute_unaudited(
        &mut self,
        module_id: &str,
        instruction: &Instruction,
        act_budget: u32,
    ) -> Result<ActResult, DispatchError> {
        let module = self
            .modules
            .get_mut(module_id)
            .ok_or_else(|| DispatchError::UnknownModule(module_id.to_string()))?;
        Ok(match catch_unwind(AssertUnwindSafe(|| module.execute(instruction, act_budget))) {
            Ok(r) => r,
            Err(payload) => ActResult {
                ok: false,
                output_delta: None,
                residue: 0.0,
                completed: true,
                error: Some(panic_message(&payload)),
            },
        })
    }

    /// The audit log, oldest first.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit
    }

    /// Empty the audit log. Act ids keep increasing (they are never reused).
    pub fn clear_audit_log(&mut self) {
        self.audit.clear();
    }

    /// Register a post-dispatch hook.
    pub fn on_dispatch(&mut self, hook: impl FnMut(&AuditEntry) + Send + 'static) -> HookId {
        let id = HookId(self.next_hook);
        self.next_hook += 1;
        self.hooks.push((id, Box::new(hook)));
        id
    }

    /// Remove a hook. Returns whether it was registered.
    pub fn remove_hook(&mut self, id: HookId) -> bool {
        let before = self.hooks.len();
        self.hooks.retain(|(h, _)| *h != id);
        self.hooks.len() != before
    }

    /// Remove every hook.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "module panicked".to_string()
    }
}

/// RFC 3339 UTC timestamp with millisecond precision, matching JS
/// `new Date().toISOString()`.
pub fn rfc3339_now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    rfc3339_from_millis(d.as_millis() as i64)
}

/// Format milliseconds since the Unix epoch as `YYYY-MM-DDTHH:MM:SS.mmmZ`.
pub fn rfc3339_from_millis(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let millis = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_instants() {
        assert_eq!(rfc3339_from_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_from_millis(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(rfc3339_from_millis(1_790_553_600_123), "2026-09-28T00:00:00.123Z");
    }
}
