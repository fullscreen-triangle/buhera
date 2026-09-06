//! Where work runs.
//!
//! The gateway is a decision plane: given a unit of work and an account,
//! it picks a catalyst to execute it on. The interesting case is the one
//! the whole design exists for — the scientist's office machine is the
//! preferred target, and when it is asleep the session must keep working
//! at reduced capability rather than failing.
//!
//! So routing is: prefer a live catalyst that has the capability, else
//! fall back to the gateway itself, else refuse. Refusal only happens
//! when the gateway genuinely cannot do the work, which is a real
//! outcome and not an error to paper over.

use crate::store::Catalyst;

/// How long after its last heartbeat a catalyst is still considered live.
///
/// The relay touches `last_seen` on a heartbeat interval; this must be a
/// comfortable multiple of it so one dropped packet does not read as a
/// sleeping machine. Sixty seconds against a fifteen-second heartbeat
/// tolerates three consecutive misses.
pub const LIVENESS_WINDOW_SECS: i64 = 60;

/// Where a unit of work should execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// Relay to a named machine of the user's that is currently connected.
    Catalyst {
        /// The catalyst's name within the account.
        name: String,
    },
    /// Run on the gateway itself — the degraded path taken when no live
    /// catalyst can serve the request.
    Gateway {
        /// Why we fell back, for surfacing in the UI. The user should be
        /// told their office machine is asleep rather than silently
        /// getting different results.
        reason: FallbackReason,
    },
}

/// Why work landed on the gateway instead of the user's own machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// The account has no machines registered at all.
    NoCatalystsRegistered,
    /// Machines are registered but none is currently connected.
    AllCatalystsAsleep,
    /// A live machine exists but none advertises the needed capability.
    NoCapableCatalyst,
    /// The caller explicitly asked for the gateway.
    Requested,
}

impl FallbackReason {
    /// A short phrase suitable for showing to the user.
    pub fn describe(self) -> &'static str {
        match self {
            FallbackReason::NoCatalystsRegistered => {
                "no machines paired — running on the gateway"
            }
            FallbackReason::AllCatalystsAsleep => {
                "your machines are offline — running on the gateway"
            }
            FallbackReason::NoCapableCatalyst => {
                "no paired machine offers this capability — running on the gateway"
            }
            FallbackReason::Requested => "running on the gateway as requested",
        }
    }
}

/// Why no route could be produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RouteError {
    /// Neither a catalyst nor the gateway can do this work.
    #[error("no route: the gateway cannot perform {capability:?} and no paired machine is available")]
    Unsatisfiable {
        /// The capability that could not be placed.
        capability: String,
    },
}

/// What the gateway itself can execute.
///
/// Deliberately narrow. The gateway runs the kernel and vaHera because
/// those are self-contained and cheap; it does **not** claim capabilities
/// that would mean reading a user's files, because the gateway has no
/// access to them and never should. A request needing those is honestly
/// unsatisfiable when the office machine is asleep — reporting that is
/// better than silently returning an answer computed against nothing.
pub const GATEWAY_CAPABILITIES: &[&str] = &["kernel", "vahera"];

/// Is `capability` something the gateway can do on its own?
pub fn gateway_can(capability: &str) -> bool {
    GATEWAY_CAPABILITIES.contains(&capability)
}

/// A catalyst counts as live if it heartbeat within the window.
pub fn is_live(catalyst: &Catalyst, now: i64) -> bool {
    match catalyst.last_seen {
        Some(seen) => now - seen <= LIVENESS_WINDOW_SECS,
        None => false,
    }
}

/// Choose where to run work needing `capability`.
///
/// `prefer` names a specific machine; when it is given and that machine
/// is live and capable, it wins. Otherwise the most recently seen live
/// capable catalyst is chosen — recency being the best available proxy
/// for "the machine the user is actually sitting at".
pub fn route(
    catalysts: &[Catalyst],
    capability: &str,
    prefer: Option<&str>,
    now: i64,
) -> Result<Route, RouteError> {
    let live: Vec<&Catalyst> = catalysts.iter().filter(|c| is_live(c, now)).collect();

    let capable = |c: &Catalyst| c.capabilities.iter().any(|cap| cap == capability);

    // An explicit request for the gateway is honoured when it can comply.
    if prefer == Some("gateway") {
        return if gateway_can(capability) {
            Ok(Route::Gateway {
                reason: FallbackReason::Requested,
            })
        } else {
            Err(RouteError::Unsatisfiable {
                capability: capability.to_string(),
            })
        };
    }

    // A named machine, if it can serve.
    if let Some(name) = prefer {
        if let Some(c) = live.iter().find(|c| c.name == name && capable(c)) {
            return Ok(Route::Catalyst {
                name: c.name.clone(),
            });
        }
    }

    // Otherwise the most recently seen capable live machine.
    let best = live
        .iter()
        .filter(|c| capable(c))
        .max_by_key(|c| c.last_seen.unwrap_or(i64::MIN));

    if let Some(c) = best {
        return Ok(Route::Catalyst {
            name: c.name.clone(),
        });
    }

    // Nothing of the user's can serve it. Say precisely why, then decide
    // whether the gateway can stand in.
    let reason = if catalysts.is_empty() {
        FallbackReason::NoCatalystsRegistered
    } else if live.is_empty() {
        FallbackReason::AllCatalystsAsleep
    } else {
        FallbackReason::NoCapableCatalyst
    };

    if gateway_can(capability) {
        Ok(Route::Gateway { reason })
    } else {
        Err(RouteError::Unsatisfiable {
            capability: capability.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_800_000_000;

    fn cat(name: &str, caps: &[&str], last_seen: Option<i64>) -> Catalyst {
        Catalyst {
            name: name.to_string(),
            account_id: "acct".to_string(),
            capabilities: caps.iter().map(|s| s.to_string()).collect(),
            last_seen,
            created_at: NOW,
        }
    }

    #[test]
    fn prefers_a_live_machine_over_the_gateway() {
        let cs = vec![cat("office", &["kernel", "spraypaint"], Some(NOW - 5))];
        assert_eq!(
            route(&cs, "kernel", None, NOW).unwrap(),
            Route::Catalyst {
                name: "office".into()
            }
        );
    }

    #[test]
    fn falls_back_when_the_office_machine_is_asleep() {
        // The core scenario: scientist is in the lab, office box is off.
        // A kernel session must still work.
        let cs = vec![cat("office", &["kernel"], Some(NOW - 3600))];
        assert_eq!(
            route(&cs, "kernel", None, NOW).unwrap(),
            Route::Gateway {
                reason: FallbackReason::AllCatalystsAsleep
            }
        );
    }

    #[test]
    fn asleep_plus_local_only_capability_is_honestly_unsatisfiable() {
        // spraypaint reads the user's files. With the machine asleep there
        // is no honest answer, so we refuse rather than invent one.
        let cs = vec![cat("office", &["spraypaint"], Some(NOW - 3600))];
        assert_eq!(
            route(&cs, "spraypaint", None, NOW),
            Err(RouteError::Unsatisfiable {
                capability: "spraypaint".into()
            })
        );
    }

    #[test]
    fn distinguishes_never_paired_from_asleep() {
        assert_eq!(
            route(&[], "kernel", None, NOW).unwrap(),
            Route::Gateway {
                reason: FallbackReason::NoCatalystsRegistered
            }
        );
    }

    #[test]
    fn live_but_incapable_is_its_own_reason() {
        let cs = vec![cat("office", &["gpu"], Some(NOW - 5))];
        assert_eq!(
            route(&cs, "kernel", None, NOW).unwrap(),
            Route::Gateway {
                reason: FallbackReason::NoCapableCatalyst
            }
        );
    }

    #[test]
    fn honours_an_explicit_machine() {
        let cs = vec![
            cat("office", &["kernel"], Some(NOW - 30)),
            cat("laptop", &["kernel"], Some(NOW - 1)),
        ];
        assert_eq!(
            route(&cs, "kernel", Some("laptop"), NOW).unwrap(),
            Route::Catalyst {
                name: "laptop".into()
            }
        );
    }

    #[test]
    fn picks_the_most_recently_seen_when_unspecified() {
        let cs = vec![
            cat("office", &["kernel"], Some(NOW - 30)),
            cat("laptop", &["kernel"], Some(NOW - 1)),
        ];
        assert_eq!(
            route(&cs, "kernel", None, NOW).unwrap(),
            Route::Catalyst {
                name: "laptop".into()
            }
        );
    }

    #[test]
    fn a_named_but_sleeping_machine_does_not_win() {
        // Naming a machine expresses a preference, not a guarantee. If it
        // is asleep the request still has to go somewhere.
        let cs = vec![
            cat("office", &["kernel"], Some(NOW - 3600)),
            cat("laptop", &["kernel"], Some(NOW - 1)),
        ];
        assert_eq!(
            route(&cs, "kernel", Some("office"), NOW).unwrap(),
            Route::Catalyst {
                name: "laptop".into()
            }
        );
    }

    #[test]
    fn liveness_boundary_is_where_it_says_it_is() {
        let just_inside = cat("a", &["kernel"], Some(NOW - LIVENESS_WINDOW_SECS));
        let just_outside = cat("b", &["kernel"], Some(NOW - LIVENESS_WINDOW_SECS - 1));
        assert!(is_live(&just_inside, NOW));
        assert!(!is_live(&just_outside, NOW));
    }

    #[test]
    fn never_connected_is_not_live() {
        assert!(!is_live(&cat("a", &["kernel"], None), NOW));
    }

    #[test]
    fn gateway_can_be_demanded_explicitly() {
        let cs = vec![cat("office", &["kernel"], Some(NOW - 1))];
        assert_eq!(
            route(&cs, "kernel", Some("gateway"), NOW).unwrap(),
            Route::Gateway {
                reason: FallbackReason::Requested
            }
        );
    }

    #[test]
    fn gateway_refuses_what_it_cannot_actually_do() {
        assert_eq!(
            route(&[], "spraypaint", Some("gateway"), NOW),
            Err(RouteError::Unsatisfiable {
                capability: "spraypaint".into()
            })
        );
    }
}
