//! Verified bearer tokens.
//!
//! The olduvai deployment on the same host documents, in its
//! `participant.rs`, that its token is "not decoded … used whole, as an
//! opaque key. This is not authentication and does not claim to be."
//! That is defensible for a service bound to loopback, where the only
//! caller is a trusted BFF.
//!
//! The gateway cannot make that trade. Its entire purpose is to accept
//! connections from a CLI running on the user's own machine, across the
//! public internet — so the token *is* the authentication boundary, and
//! an unverified opaque string would mean anyone able to reach the port
//! can act as any user by guessing.
//!
//! So: HMAC-SHA256 over a compact claims payload, with an explicit
//! expiry and an audience tag separating the two kinds of credential.
//! Verification is constant-time. No JWT dependency — the claim set is
//! fixed and small, and hand-rolling the parse keeps the trusted
//! surface auditable.
//!
//! Wire format (all ASCII, `.`-separated, URL-safe base64, no padding):
//!
//! ```text
//! v1.<payload-b64>.<mac-b64>
//! ```
//!
//! where `payload` is `<aud>:<subject>:<issued-unix>:<expires-unix>:<nonce>`.
//! The nonce makes two tokens minted in the same second distinct, so a
//! revocation list can name one without naming the other.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// What a token is allowed to be used for.
///
/// A session token authenticates a browser; a catalyst token
/// authenticates a machine dialing in to offer compute. They are
/// deliberately not interchangeable: a leaked catalyst token must not
/// grant access to the account's web session, and vice versa. The
/// audience is inside the signed payload, so it cannot be edited by the
/// holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Audience {
    /// Issued at login, presented by the browser.
    Session,
    /// Issued when a machine is registered, presented by the CLI when
    /// it dials the relay.
    Catalyst,
}

impl Audience {
    /// The wire tag. Must never collide with the field separator.
    pub fn as_str(self) -> &'static str {
        match self {
            Audience::Session => "ses",
            Audience::Catalyst => "cat",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "ses" => Some(Audience::Session),
            "cat" => Some(Audience::Catalyst),
            _ => None,
        }
    }
}

/// A verified token's contents.
///
/// Only ever constructed by [`Signer::verify`] — holding one is proof
/// the signature checked out and the clock had not passed `expires_at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claims {
    /// What this token may be used for.
    pub audience: Audience,
    /// Account id the token speaks for.
    pub subject: String,
    /// Unix seconds at issue.
    pub issued_at: i64,
    /// Unix seconds after which the token is refused.
    pub expires_at: i64,
    /// Random per-token id, so one token can be revoked without
    /// revoking every token minted for the same subject in the same
    /// second.
    pub nonce: String,
}

/// Why a token was refused.
///
/// Deliberately coarse at the boundary: callers should map every
/// variant to the same 401 with no detail, so the response does not
/// tell an attacker whether a token was well-formed, expired, or
/// forged. The distinction exists for logs, not for clients.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TokenError {
    /// Structure did not match `v1.<payload>.<mac>`.
    #[error("malformed token")]
    Malformed,
    /// Signature did not verify under the server key.
    #[error("bad signature")]
    BadSignature,
    /// `expires_at` is in the past.
    #[error("token expired")]
    Expired,
    /// Verified, but minted for a different purpose.
    #[error("wrong audience")]
    WrongAudience,
}

/// The server's signing key.
///
/// Generated on the host and read from the environment — never
/// committed, never transmitted. Dropping the key invalidates every
/// token in circulation, which is the blunt revocation lever.
#[derive(Clone)]
pub struct Signer {
    key: Vec<u8>,
}

impl std::fmt::Debug for Signer {
    /// Redacted: a key that prints is a key that ends up in a log.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signer").field("key", &"<redacted>").finish()
    }
}

impl Signer {
    /// Build from raw key bytes.
    ///
    /// Rejects keys under 32 bytes: a short HMAC key is the one
    /// configuration mistake that silently weakens everything else here,
    /// so it fails loudly at startup rather than serving traffic.
    pub fn new(key: impl Into<Vec<u8>>) -> Result<Self, String> {
        let key = key.into();
        if key.len() < 32 {
            return Err(format!(
                "signing key must be at least 32 bytes, got {}",
                key.len()
            ));
        }
        Ok(Self { key })
    }

    /// Generate a fresh random key, for first boot.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut key = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        Self { key }
    }

    /// The key as URL-safe base64, for writing to a config file once.
    pub fn to_base64(&self) -> String {
        URL_SAFE_NO_PAD.encode(&self.key)
    }

    /// Parse a key previously produced by [`Signer::to_base64`].
    pub fn from_base64(s: &str) -> Result<Self, String> {
        let raw = URL_SAFE_NO_PAD
            .decode(s.trim())
            .map_err(|e| format!("signing key is not valid base64: {e}"))?;
        Self::new(raw)
    }

    fn mac(&self, payload: &str) -> Vec<u8> {
        let mut m = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC accepts keys of any length; length was checked in Signer::new");
        m.update(payload.as_bytes());
        m.finalize().into_bytes().to_vec()
    }

    /// Mint a token valid for `ttl_secs` from `now`.
    ///
    /// `now` is passed in rather than read from the clock so the expiry
    /// logic is testable without sleeping.
    pub fn mint(&self, audience: Audience, subject: &str, now: i64, ttl_secs: i64) -> String {
        let nonce = {
            use rand::RngCore;
            let mut b = [0u8; 12];
            rand::thread_rng().fill_bytes(&mut b);
            URL_SAFE_NO_PAD.encode(b)
        };
        let payload = format!(
            "{}:{}:{}:{}:{}",
            audience.as_str(),
            subject,
            now,
            now + ttl_secs,
            nonce
        );
        let mac = self.mac(&payload);
        format!(
            "v1.{}.{}",
            URL_SAFE_NO_PAD.encode(payload.as_bytes()),
            URL_SAFE_NO_PAD.encode(mac)
        )
    }

    /// Verify `token`, requiring it to carry `expected` audience.
    ///
    /// Order matters: the signature is checked *before* the payload is
    /// trusted for anything, including expiry. Reading a claim out of an
    /// unverified payload and acting on it is the classic way this kind
    /// of code goes wrong.
    pub fn verify(&self, token: &str, expected: Audience, now: i64) -> Result<Claims, TokenError> {
        let mut parts = token.split('.');
        let (Some(version), Some(payload_b64), Some(mac_b64), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(TokenError::Malformed);
        };
        if version != "v1" {
            return Err(TokenError::Malformed);
        }

        let payload_bytes = URL_SAFE_NO_PAD
            .decode(payload_b64)
            .map_err(|_| TokenError::Malformed)?;
        let presented_mac = URL_SAFE_NO_PAD
            .decode(mac_b64)
            .map_err(|_| TokenError::Malformed)?;
        let payload = std::str::from_utf8(&payload_bytes).map_err(|_| TokenError::Malformed)?;

        // Constant-time compare: a byte-by-byte `==` leaks how much of a
        // forged MAC was correct, which is enough to forge one given
        // enough attempts.
        let expected_mac = self.mac(payload);
        if expected_mac.ct_eq(&presented_mac).unwrap_u8() != 1 {
            return Err(TokenError::BadSignature);
        }

        // Only now is the payload trustworthy.
        let fields: Vec<&str> = payload.split(':').collect();
        let [aud, subject, issued, expires, nonce] = fields[..] else {
            return Err(TokenError::Malformed);
        };
        let audience = Audience::parse(aud).ok_or(TokenError::Malformed)?;
        let issued_at: i64 = issued.parse().map_err(|_| TokenError::Malformed)?;
        let expires_at: i64 = expires.parse().map_err(|_| TokenError::Malformed)?;

        if now >= expires_at {
            return Err(TokenError::Expired);
        }
        if audience != expected {
            return Err(TokenError::WrongAudience);
        }

        Ok(Claims {
            audience,
            subject: subject.to_string(),
            issued_at,
            expires_at,
            nonce: nonce.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_800_000_000;
    const HOUR: i64 = 3600;

    fn signer() -> Signer {
        Signer::new(vec![7u8; 32]).expect("32-byte key is accepted")
    }

    #[test]
    fn round_trips() {
        let s = signer();
        let t = s.mint(Audience::Session, "user-1", NOW, HOUR);
        let c = s
            .verify(&t, Audience::Session, NOW)
            .expect("fresh token verifies");
        assert_eq!(c.subject, "user-1");
        assert_eq!(c.audience, Audience::Session);
        assert_eq!(c.expires_at, NOW + HOUR);
    }

    #[test]
    fn rejects_expired() {
        let s = signer();
        let t = s.mint(Audience::Session, "user-1", NOW, HOUR);
        // Exactly at expiry is already too late.
        assert_eq!(
            s.verify(&t, Audience::Session, NOW + HOUR),
            Err(TokenError::Expired)
        );
    }

    #[test]
    fn rejects_foreign_key() {
        let a = signer();
        let b = Signer::new(vec![9u8; 32]).unwrap();
        let t = a.mint(Audience::Session, "user-1", NOW, HOUR);
        assert_eq!(
            b.verify(&t, Audience::Session, NOW),
            Err(TokenError::BadSignature)
        );
    }

    #[test]
    fn rejects_tampered_subject() {
        // The attack the olduvai note describes: edit the identity and
        // present it. Here the MAC covers the subject, so it fails.
        let s = signer();
        let t = s.mint(Audience::Session, "user-1", NOW, HOUR);
        let payload = format!("ses:{}:{}:{}:{}", "admin", NOW, NOW + HOUR, "AAAAAAAAAAAA");
        let forged = format!(
            "v1.{}.{}",
            URL_SAFE_NO_PAD.encode(payload.as_bytes()),
            t.split('.').nth(2).unwrap()
        );
        assert_eq!(
            s.verify(&forged, Audience::Session, NOW),
            Err(TokenError::BadSignature)
        );
    }

    #[test]
    fn audiences_do_not_cross() {
        // A catalyst token must not open a browser session.
        let s = signer();
        let t = s.mint(Audience::Catalyst, "user-1", NOW, HOUR);
        assert_eq!(
            s.verify(&t, Audience::Session, NOW),
            Err(TokenError::WrongAudience)
        );
    }

    #[test]
    fn rejects_malformed() {
        let s = signer();
        for bad in ["", "v1", "v1.a", "v2.a.b", "v1.a.b.c", "not-a-token"] {
            assert_eq!(
                s.verify(bad, Audience::Session, NOW),
                Err(TokenError::Malformed),
                "expected {bad:?} to be rejected as malformed"
            );
        }
    }

    #[test]
    fn short_keys_are_refused() {
        assert!(Signer::new(vec![1u8; 31]).is_err());
        assert!(Signer::new(vec![1u8; 32]).is_ok());
    }

    #[test]
    fn key_survives_base64_round_trip() {
        let s = Signer::generate();
        let restored = Signer::from_base64(&s.to_base64()).expect("round trip");
        let t = s.mint(Audience::Session, "u", NOW, HOUR);
        assert!(restored.verify(&t, Audience::Session, NOW).is_ok());
    }
}
