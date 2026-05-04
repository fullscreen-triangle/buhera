//! [`BuheraKernel`]: the dispatch wrapper that composes the five subsystems
//! over a [`purpose_operations::Executor`].
//!
//! ```text
//!     dispatch(fragment)
//!         │
//!         ├─ PVE.validate(fragment)       (Phase 2: Three-Route Equivalence)
//!         ├─ PSS.order(fragment)          (Phase 4: critical-slowing scheduling)
//!         ├─ CMM.lookup_fragment(fragment) (Phase 3: cache extinction)
//!         │       │  miss
//!         │       ▼
//!         ├─ Executor.execute(fragment)
//!         ├─ CMM.insert(...)
//!         ├─ PSS.observe_completion(fragment)
//!         ├─ TEM.observe(event)           (Phase 5: MTIC + monotone log)
//!         └─ publish DispatchEvent
//! ```
//!
//! In Phase 1 every subsystem is a no-op, so the call collapses to a
//! straight pass-through over the bare `Executor`. The dispatch path,
//! ordering, and event surface are nevertheless permanent.

use std::sync::Arc;
use std::time::Instant;

use purpose_core::{Error, VaHera, Value};
use purpose_operations::{Executor, OperationRegistry};

use crate::event::DispatchEvent;
use crate::observable::Observable;
use crate::subsystem::cmm::{Cmm, Lookup, NoopCmm};
use crate::subsystem::dic::{Dic, NoopDic};
use crate::subsystem::pss::{NoopPss, Pss};
use crate::subsystem::pve::{NoopPve, Pve};
use crate::subsystem::tem::{NoopTem, Tem};

/// The Buhera Kernel — dispatch front-end for the Purpose framework.
///
/// Construct with [`BuheraKernelBuilder`]; dispatch with [`dispatch`].
/// Subscribe to dispatch events with [`events`].
///
/// [`dispatch`]: BuheraKernel::dispatch
/// [`events`]: BuheraKernel::events
pub struct BuheraKernel {
    executor: Executor,
    pve: Arc<dyn Pve>,
    cmm: Arc<dyn Cmm>,
    pss: Arc<dyn Pss>,
    #[allow(dead_code)] // wired up in Phase 6
    dic: Arc<dyn Dic>,
    tem: Arc<dyn Tem>,
    events: Observable<DispatchEvent>,
}

impl BuheraKernel {
    /// Start a kernel build. The default subsystems are the Phase-1 no-op
    /// stubs; replace them via the builder methods as later phases ship.
    pub fn builder(registry: OperationRegistry) -> BuheraKernelBuilder {
        BuheraKernelBuilder::new(registry)
    }

    /// Dispatch a fragment.
    ///
    /// Wraps [`Executor::execute`] with the five subsystems. In Phase 1 the
    /// result is bit-for-bit identical to a direct executor call; the only
    /// observable differences are (a) the dispatch event published on
    /// [`events`](Self::events) and (b) the fact that every subsystem
    /// receives its hooks in turn.
    pub async fn dispatch(&self, fragment: &VaHera) -> Result<Value, Error> {
        let start = Instant::now();

        // 1. Pre-dispatch validation. In Phase 1 always Ok(()).
        self.pve.validate(fragment).await?;

        // 2. PSS ordering hook. In Phase 1 always returns Order::Now and we
        //    proceed directly. Phase 4 will introduce a queue.
        let _order = self.pss.order(fragment).await;

        // 3. CMM lookup. In Phase 1 always Miss.
        let cache_hit;
        let result = match self.cmm.lookup_fragment(fragment).await {
            Lookup::Hit(value) => {
                cache_hit = true;
                Ok(value)
            }
            Lookup::Miss => {
                cache_hit = false;
                self.executor.execute(fragment).await
            }
        };

        // 4. Insert into the CMM on success. Phase 1 drops the insertion.
        if let (false, Ok(ref value)) = (cache_hit, &result) {
            insert_top_level_call(self.cmm.as_ref(), fragment, value).await;
        }

        // 5. Notify PSS that this fragment finished. Phase 1: no-op.
        self.pss.observe_completion(fragment).await;

        // 6. Build and publish the dispatch event.
        let mut event = DispatchEvent::new(top_level_op(fragment), start.elapsed(), result.is_ok());
        event.cache_hit = cache_hit;
        self.tem.observe(&event).await;
        self.events.publish(event);

        result
    }

    /// Subscribe to dispatch events. The returned [`Observable`] is a clone
    /// of the kernel's internal handle; subscribers do not back-pressure
    /// the dispatch path.
    pub fn events(&self) -> Observable<DispatchEvent> {
        self.events.clone()
    }

    /// Borrow the underlying executor's registry. Useful for the CLI to
    /// list operations without poking through kernel internals.
    pub fn registry(&self) -> &OperationRegistry {
        self.executor.registry()
    }
}

/// Builder for [`BuheraKernel`]. Replace any subsystem with a non-no-op
/// implementation before calling [`build`](Self::build).
pub struct BuheraKernelBuilder {
    registry: OperationRegistry,
    pve: Arc<dyn Pve>,
    cmm: Arc<dyn Cmm>,
    pss: Arc<dyn Pss>,
    dic: Arc<dyn Dic>,
    tem: Arc<dyn Tem>,
    event_capacity: usize,
}

impl BuheraKernelBuilder {
    /// Construct a builder with all-no-op subsystems and a default 256-slot
    /// event channel.
    pub fn new(registry: OperationRegistry) -> Self {
        Self {
            registry,
            pve: Arc::new(NoopPve),
            cmm: Arc::new(NoopCmm),
            pss: Arc::new(NoopPss),
            dic: Arc::new(NoopDic),
            tem: Arc::new(NoopTem),
            event_capacity: 256,
        }
    }

    /// Install a custom PVE.
    pub fn with_pve(mut self, pve: Arc<dyn Pve>) -> Self {
        self.pve = pve;
        self
    }

    /// Install a custom CMM.
    pub fn with_cmm(mut self, cmm: Arc<dyn Cmm>) -> Self {
        self.cmm = cmm;
        self
    }

    /// Install a custom PSS.
    pub fn with_pss(mut self, pss: Arc<dyn Pss>) -> Self {
        self.pss = pss;
        self
    }

    /// Install a custom DIC.
    pub fn with_dic(mut self, dic: Arc<dyn Dic>) -> Self {
        self.dic = dic;
        self
    }

    /// Install a custom TEM.
    pub fn with_tem(mut self, tem: Arc<dyn Tem>) -> Self {
        self.tem = tem;
        self
    }

    /// Override the broadcast-channel capacity for dispatch events.
    pub fn event_capacity(mut self, capacity: usize) -> Self {
        self.event_capacity = capacity;
        self
    }

    /// Finish construction.
    pub fn build(self) -> BuheraKernel {
        BuheraKernel {
            executor: Executor::new(self.registry),
            pve: self.pve,
            cmm: self.cmm,
            pss: self.pss,
            dic: self.dic,
            tem: self.tem,
            events: Observable::new(self.event_capacity),
        }
    }
}

fn top_level_op(fragment: &VaHera) -> Option<String> {
    match fragment {
        VaHera::Call { op, .. } => Some(op.clone()),
        VaHera::Compose(parts) => parts.last().and_then(top_level_op),
        VaHera::Literal(_) | VaHera::Hole(_) => None,
    }
}

async fn insert_top_level_call(cmm: &dyn Cmm, fragment: &VaHera, value: &Value) {
    if let VaHera::Call { op, args } = fragment {
        // Args may contain VaHera fragments rather than fully-resolved
        // Values; Phase 3 will canonicalise these properly. For Phase 1
        // we only insert when every arg is a Literal so the cache key
        // remains hashable. Anything richer is dropped silently.
        let mut resolved = std::collections::BTreeMap::new();
        for (k, frag) in args.iter() {
            if let VaHera::Literal(v) = frag {
                resolved.insert(k.clone(), v.clone());
            } else {
                return;
            }
        }
        cmm.insert(op, &resolved, value).await;
    }
}
