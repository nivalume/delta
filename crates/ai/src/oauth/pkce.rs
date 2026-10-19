//! PKCE (RFC 7636) verifier and S256 challenge. Port of upstream `auth/oauth/pkce.ts`.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

/// A PKCE pair: send `challenge` (with `code_challenge_method=S256`) in the authorization
/// request and `verifier` in the token exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pkce {
    /// 32 random bytes, base64url without padding (43 characters).
    pub verifier: String,
    /// `base64url(SHA-256(verifier))` without padding.
    pub challenge: String,
}

/// Base64url without padding (`+` → `-`, `/` → `_`, no `=`).
pub fn base64url_encode(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

/// The S256 challenge of `verifier`: `base64url(SHA-256(verifier))`.
pub fn pkce_challenge(verifier: &str) -> String {
    base64url_encode(&Sha256::digest(verifier.as_bytes()))
}

/// Generate a PKCE verifier from 32 bytes of OS randomness and its S256 challenge.
///
/// Upstream `generatePKCE()` is async because Web Crypto is; this is synchronous.
pub fn generate_pkce() -> Pkce {
    let mut bytes = [0u8; 32];
    // The OS random source only fails on unsupported platforms; there is no fallback.
    getrandom::fill(&mut bytes).expect("OS random number generator is unavailable");
    let verifier = base64url_encode(&bytes);
    let challenge = pkce_challenge(&verifier);
    Pkce { verifier, challenge }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s256_challenge_is_base64url_sha256() {
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r7wW1gFWFOEjXk"),
            "bwWFMyPfdG9qreDhH2lmftFx_dFeLDalzcT1gb_j68g"
        );
    }

    #[test]
    fn generates_distinct_base64url_verifiers() {
        let a = generate_pkce();
        let b = generate_pkce();
        assert_ne!(a.verifier, b.verifier);
        assert_eq!(a.verifier.len(), 43);
        assert!(
            a.verifier
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        );
        assert_eq!(a.challenge, pkce_challenge(&a.verifier));
    }
}
