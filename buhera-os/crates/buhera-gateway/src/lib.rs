//! Buhera gateway.
//!
//! The gateway is the piece that lets a user's Buhera instance follow them
//! between machines. It holds the account, the session, and the roster of
//! machines that account can compute on; it does **not** hold the user's
//! kernel contents — those live on the user's own machine, reached through
//! the relay.
//!
//! The shape, concretely: a scientist pairs their office workstation once,
//! which registers it as a catalyst. From the lab they log in through the
//! web app; the gateway routes their work back to the office machine over
//! a connection that machine dialed out. When the office machine is asleep
//! the session survives at reduced capability, running whatever the gateway
//! itself can do.
//!
//! Modules:
//!
//! * [`token`]  — signed, expiring bearer tokens for both browser sessions
//!                and machine registrations.
//! * [`store`]  — durable accounts and catalyst roster (SQLite).
//! * [`router`] — the placement decision, including the degraded path.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod router;
pub mod store;
pub mod token;

/// Current wall-clock in Unix seconds.
///
/// Every module here takes `now` as a parameter rather than reading the
/// clock internally, so expiry and liveness are testable without sleeping.
/// This is the one place the real clock is consulted.
pub fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
