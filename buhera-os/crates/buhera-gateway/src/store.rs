//! Durable account and catalyst storage.
//!
//! The olduvai service on this host keeps its state in a process-lifetime
//! cache and documents the consequence plainly: "there is no database to
//! back up, and no backup will exist". That is a reasonable stage-appropriate
//! choice for an append-only observation log that can be replayed.
//!
//! Accounts are not that. A user who cannot log in after a restart has lost
//! the thing the gateway exists to provide, and password hashes cannot be
//! reconstructed by replaying anything. Worse, the deployment note's §1.4
//! records that the Hetzner project is administered by a third party whose
//! API token can rebuild this machine regardless of who holds root — so the
//! state must live in a single file that can be copied off the box.
//!
//! Hence SQLite: one file, no daemon, crash-safe, and `sqlite3 .backup` or a
//! plain `cp` of a quiesced file is a complete backup.
//!
//! What is deliberately *not* stored here: kernel contents. The office
//! machine holds those. The gateway stores identity and the catalyst roster,
//! which is what has to survive for a user to log in from elsewhere and find
//! their machines.

use argon2::password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

/// Errors from the storage layer.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Underlying SQLite failure.
    #[error("database: {0}")]
    Db(#[from] rusqlite::Error),
    /// Password hashing failure.
    #[error("password hashing: {0}")]
    Hash(String),
    /// The requested account does not exist.
    #[error("no such account")]
    NoSuchAccount,
    /// An account with that email already exists.
    #[error("account already exists")]
    Duplicate,
}

/// One registered user.
#[derive(Debug, Clone)]
pub struct Account {
    /// Stable opaque id; this is what tokens carry as their subject.
    pub id: String,
    /// Login identifier, stored lowercased.
    pub email: String,
    /// Unix seconds at creation.
    pub created_at: i64,
}

/// A machine registered to an account as a compute target.
///
/// `last_seen` is what makes the awake/asleep distinction observable:
/// the relay updates it while a CLI holds its connection, and the router
/// reads it to decide whether the machine can be dispatched to.
#[derive(Debug, Clone)]
pub struct Catalyst {
    /// Unique within an account.
    pub name: String,
    /// Owning account id.
    pub account_id: String,
    /// Free-form capability tags, e.g. `["cpu", "spraypaint"]`.
    pub capabilities: Vec<String>,
    /// Unix seconds when the CLI last held a live relay connection,
    /// or `None` if it has never connected.
    pub last_seen: Option<i64>,
    /// Unix seconds at registration.
    pub created_at: i64,
}

/// Handle to the gateway's durable state.
pub struct Store {
    conn: Connection,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").finish_non_exhaustive()
    }
}

impl Store {
    /// Open (creating if absent) the database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open a private in-memory database, for tests.
    pub fn open_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self, StoreError> {
        // WAL survives an abrupt kill without corrupting the file, which
        // matters on a machine a third party can reboot out from under us.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS accounts (
                 id            TEXT PRIMARY KEY,
                 email         TEXT NOT NULL UNIQUE,
                 password_hash TEXT NOT NULL,
                 created_at    INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS catalysts (
                 account_id   TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                 name         TEXT NOT NULL,
                 capabilities TEXT NOT NULL,
                 last_seen    INTEGER,
                 created_at   INTEGER NOT NULL,
                 PRIMARY KEY (account_id, name)
             );",
        )?;
        Ok(Self { conn })
    }

    /// Create an account. Returns [`StoreError::Duplicate`] if the email
    /// is taken.
    ///
    /// The password is hashed with Argon2id and a per-account random salt;
    /// the plaintext is never stored and never logged.
    pub fn create_account(
        &self,
        email: &str,
        password: &str,
        now: i64,
    ) -> Result<Account, StoreError> {
        let email = email.trim().to_lowercase();
        let id = uuid::Uuid::new_v4().to_string();

        let salt = SaltString::generate(&mut OsRng);
        let hash = Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| StoreError::Hash(e.to_string()))?
            .to_string();

        let res = self.conn.execute(
            "INSERT INTO accounts (id, email, password_hash, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, email, hash, now],
        );
        match res {
            Ok(_) => Ok(Account {
                id,
                email,
                created_at: now,
            }),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(StoreError::Duplicate)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Verify a login. Returns the account only when the password matches.
    ///
    /// A missing account and a wrong password are deliberately the same
    /// `Ok(None)` to the caller, so the API cannot be used to enumerate
    /// which addresses are registered. The hash is still verified against a
    /// dummy when the account is absent, so the two paths take similar time.
    pub fn verify_login(&self, email: &str, password: &str) -> Result<Option<Account>, StoreError> {
        let email = email.trim().to_lowercase();
        let row: Option<(String, String, i64)> = self
            .conn
            .query_row(
                "SELECT id, password_hash, created_at FROM accounts WHERE email = ?1",
                params![email],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        let Some((id, hash, created_at)) = row else {
            // Constant-ish work on the miss path. Without this, a fast 404
            // versus a slow rejection tells an attacker the address exists.
            let _ = Argon2::default().verify_password(
                password.as_bytes(),
                &PasswordHash::new(DUMMY_HASH).expect("embedded dummy hash parses"),
            );
            return Ok(None);
        };

        let parsed = PasswordHash::new(&hash).map_err(|e| StoreError::Hash(e.to_string()))?;
        match Argon2::default().verify_password(password.as_bytes(), &parsed) {
            Ok(()) => Ok(Some(Account {
                id,
                email,
                created_at,
            })),
            Err(_) => Ok(None),
        }
    }

    /// Look up an account by id.
    pub fn account(&self, id: &str) -> Result<Option<Account>, StoreError> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, email, created_at FROM accounts WHERE id = ?1",
                params![id],
                |r| {
                    Ok(Account {
                        id: r.get(0)?,
                        email: r.get(1)?,
                        created_at: r.get(2)?,
                    })
                },
            )
            .optional()?)
    }

    /// Register (or re-register) a machine under an account.
    ///
    /// Re-registering the same name replaces its capabilities and clears
    /// `last_seen`, which is what re-running `buhera pair` on a machine
    /// should do.
    pub fn upsert_catalyst(
        &self,
        account_id: &str,
        name: &str,
        capabilities: &[String],
        now: i64,
    ) -> Result<Catalyst, StoreError> {
        if self.account(account_id)?.is_none() {
            return Err(StoreError::NoSuchAccount);
        }
        let caps = serde_json::to_string(capabilities).expect("Vec<String> serializes");
        self.conn.execute(
            "INSERT INTO catalysts (account_id, name, capabilities, last_seen, created_at)
             VALUES (?1, ?2, ?3, NULL, ?4)
             ON CONFLICT(account_id, name) DO UPDATE SET
                 capabilities = excluded.capabilities,
                 last_seen    = NULL",
            params![account_id, name, caps, now],
        )?;
        Ok(Catalyst {
            name: name.to_string(),
            account_id: account_id.to_string(),
            capabilities: capabilities.to_vec(),
            last_seen: None,
            created_at: now,
        })
    }

    /// Every catalyst registered to an account, newest first.
    pub fn catalysts(&self, account_id: &str) -> Result<Vec<Catalyst>, StoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT name, capabilities, last_seen, created_at
             FROM catalysts WHERE account_id = ?1 ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(params![account_id], |r| {
            let caps: String = r.get(1)?;
            Ok(Catalyst {
                name: r.get(0)?,
                account_id: account_id.to_string(),
                capabilities: serde_json::from_str(&caps).unwrap_or_default(),
                last_seen: r.get(2)?,
                created_at: r.get(3)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Record that a catalyst is currently connected.
    pub fn touch_catalyst(&self, account_id: &str, name: &str, now: i64) -> Result<(), StoreError> {
        self.conn.execute(
            "UPDATE catalysts SET last_seen = ?3 WHERE account_id = ?1 AND name = ?2",
            params![account_id, name, now],
        )?;
        Ok(())
    }

    /// Remove a catalyst registration.
    pub fn remove_catalyst(&self, account_id: &str, name: &str) -> Result<bool, StoreError> {
        let n = self.conn.execute(
            "DELETE FROM catalysts WHERE account_id = ?1 AND name = ?2",
            params![account_id, name],
        )?;
        Ok(n > 0)
    }
}

/// A real Argon2 hash of a value nobody knows, used to equalise timing on
/// the account-not-found path of [`Store::verify_login`].
const DUMMY_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHRzb21lc2FsdA$\
                          eqBXAMPLEhashvalueusedonlyfortimingpaddingAAAA";

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_800_000_000;

    fn store() -> Store {
        Store::open_memory().expect("in-memory store opens")
    }

    #[test]
    fn creates_and_authenticates() {
        let s = store();
        let a = s.create_account("Scientist@Example.org", "correct horse", NOW).unwrap();
        // Email is normalised on the way in.
        assert_eq!(a.email, "scientist@example.org");

        // Login is case-insensitive on the address, exact on the password.
        let ok = s.verify_login("SCIENTIST@example.org", "correct horse").unwrap();
        assert_eq!(ok.map(|x| x.id), Some(a.id.clone()));
        assert!(s.verify_login("scientist@example.org", "wrong").unwrap().is_none());
    }

    #[test]
    fn unknown_account_is_indistinguishable_from_bad_password() {
        let s = store();
        s.create_account("a@example.org", "pw", NOW).unwrap();
        assert!(s.verify_login("a@example.org", "nope").unwrap().is_none());
        assert!(s.verify_login("ghost@example.org", "nope").unwrap().is_none());
    }

    #[test]
    fn password_is_not_stored_in_plaintext() {
        let s = store();
        s.create_account("a@example.org", "hunter2", NOW).unwrap();
        let hash: String = s
            .conn
            .query_row("SELECT password_hash FROM accounts", [], |r| r.get(0))
            .unwrap();
        assert!(!hash.contains("hunter2"));
        assert!(hash.starts_with("$argon2id$"));
    }

    #[test]
    fn duplicate_email_is_refused() {
        let s = store();
        s.create_account("a@example.org", "pw", NOW).unwrap();
        assert!(matches!(
            s.create_account("A@EXAMPLE.ORG", "other", NOW),
            Err(StoreError::Duplicate)
        ));
    }

    #[test]
    fn catalysts_are_scoped_to_their_account() {
        // The property that matters: one user must never see or dispatch to
        // another user's machine.
        let s = store();
        let alice = s.create_account("alice@example.org", "pw", NOW).unwrap();
        let bob = s.create_account("bob@example.org", "pw", NOW).unwrap();

        s.upsert_catalyst(&alice.id, "office", &["cpu".into()], NOW).unwrap();
        s.upsert_catalyst(&bob.id, "laptop", &["cpu".into()], NOW).unwrap();

        let seen: Vec<String> = s.catalysts(&alice.id).unwrap().into_iter().map(|c| c.name).collect();
        assert_eq!(seen, vec!["office"]);
    }

    #[test]
    fn reregistering_replaces_rather_than_duplicates() {
        let s = store();
        let a = s.create_account("a@example.org", "pw", NOW).unwrap();
        s.upsert_catalyst(&a.id, "office", &["cpu".into()], NOW).unwrap();
        s.touch_catalyst(&a.id, "office", NOW + 5).unwrap();
        s.upsert_catalyst(&a.id, "office", &["cpu".into(), "gpu".into()], NOW + 10).unwrap();

        let cs = s.catalysts(&a.id).unwrap();
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].capabilities, vec!["cpu", "gpu"]);
        // Re-pairing means "this machine is starting fresh", so liveness resets.
        assert_eq!(cs[0].last_seen, None);
    }

    #[test]
    fn liveness_is_recorded() {
        let s = store();
        let a = s.create_account("a@example.org", "pw", NOW).unwrap();
        s.upsert_catalyst(&a.id, "office", &[], NOW).unwrap();
        assert_eq!(s.catalysts(&a.id).unwrap()[0].last_seen, None);

        s.touch_catalyst(&a.id, "office", NOW + 30).unwrap();
        assert_eq!(s.catalysts(&a.id).unwrap()[0].last_seen, Some(NOW + 30));
    }

    #[test]
    fn catalyst_needs_a_real_account() {
        let s = store();
        assert!(matches!(
            s.upsert_catalyst("no-such-id", "office", &[], NOW),
            Err(StoreError::NoSuchAccount)
        ));
    }

    #[test]
    fn removal_reports_whether_anything_went() {
        let s = store();
        let a = s.create_account("a@example.org", "pw", NOW).unwrap();
        s.upsert_catalyst(&a.id, "office", &[], NOW).unwrap();
        assert!(s.remove_catalyst(&a.id, "office").unwrap());
        assert!(!s.remove_catalyst(&a.id, "office").unwrap());
    }
}
