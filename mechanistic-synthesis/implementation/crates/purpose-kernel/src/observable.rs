//! Read-only handle over a broadcast channel of typed kernel events.
//!
//! Subsystems publish through an internal [`tokio::sync::broadcast::Sender`];
//! external consumers receive an [`Observable<T>`] and can `subscribe()` to
//! get a fresh receiver. The handle is intentionally minimal: there is no
//! way for an external consumer to publish into the kernel.
//!
//! Lagging subscribers see [`tokio::sync::broadcast::error::RecvError::Lagged`]
//! rather than blocking the kernel; this is by design — TEM and external
//! exporters must not be able to back-pressure the dispatch path.

use tokio::sync::broadcast;

/// Read-only subscription handle for kernel events of type `T`.
///
/// Cloning is cheap; every clone holds the same underlying channel handle.
#[derive(Clone)]
pub struct Observable<T: Clone + Send + 'static> {
    sender: broadcast::Sender<T>,
}

impl<T: Clone + Send + 'static> Observable<T> {
    /// Construct a fresh handle backed by a channel of the given capacity.
    /// Capacity bounds the number of events a slow subscriber can fall behind
    /// before it is forced to drop and reconnect.
    pub fn new(capacity: usize) -> Self {
        let (sender, _rx) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Subscribe to future events. Past events are not replayed.
    pub fn subscribe(&self) -> broadcast::Receiver<T> {
        self.sender.subscribe()
    }

    /// Publish an event. `Ok(n)` is the number of currently-active receivers;
    /// `Err` is returned only when there are zero subscribers, which is not
    /// itself an error condition for the kernel and is therefore swallowed
    /// at the call site.
    pub(crate) fn publish(&self, event: T) {
        // Ignoring the result: zero subscribers is normal in Phase 1.
        let _ = self.sender.send(event);
    }
}

impl<T: Clone + Send + 'static> Default for Observable<T> {
    fn default() -> Self {
        Self::new(256)
    }
}
