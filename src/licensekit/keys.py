"""License key issuance, parsing, and fingerprinting.

Key anatomy (see ``docs/KEY_FORMAT.md`` for the full spec)::

    SL1.<payload_b32>.<signature_b32>

* ``payload`` — canonical JSON: version, key id, customer email, plan,
  site limit, feature flags, issued-at and expiry as unix timestamps.
* ``signature`` — HMAC-SHA256 over ``b"survey-license-v1." + payload_bytes``,
  keyed with the vendor secret. Truncation: none — the full 32 bytes ship.

Both halves are base32 (uppercase, padding stripped) so keys survive
copy/paste, email clients, and case-mangling. The vendor secret never leaves
the vendor; engines validate with the same secret distributed alongside them
(for a stronger split, derive a verify-only sub-key — see docs/KEY_FORMAT.md).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .errors import KeyFormatError

KEY_PREFIX = "SL1"
KEY_VERSION = 1
SIGNING_DOMAIN = b"survey-license-v1"

# Plan catalogue. ``sites`` is the default site allowance (None = unlimited);
# ``features`` is the default feature-flag set. Per-key overrides are allowed
# at issuance — the plan is a template, not a straitjacket.
PLANS: Dict[str, Dict[str, object]] = {
    "trial": {
        "sites": 2,
        "features": ["monitor"],
        "description": "Time-boxed evaluation: a couple of sites, monitoring only.",
    },
    "site": {
        "sites": 1,
        "features": ["monitor", "alerts-pdf"],
        "description": "Per-site subscription: monitoring plus PDF reports for one site.",
    },
    "team": {
        "sites": 10,
        "features": ["monitor", "alerts-pdf", "alerts-email"],
        "description": "Small-team plan: up to ten sites with PDF and email delivery.",
    },
    "unlimited": {
        "sites": None,
        "features": ["monitor", "alerts-pdf", "alerts-email"],
        "description": "Seat/organization plan: unlimited sites, every feature.",
    },
}

#: Feature flags understood across the suite. Engines should only gate on
#: flags listed here so entitlements stay comparable between products.
KNOWN_FEATURES = ("monitor", "alerts-pdf", "alerts-email")


def _b32encode(raw: bytes) -> str:
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _b32decode(text: str) -> bytes:
    s = text.strip().upper()
    if not s or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in s):
        raise KeyFormatError(f"not valid base32: {text[:24]!r}…")
    try:
        return base64.b32decode(s + "=" * (-len(s) % 8))
    except Exception as exc:
        raise KeyFormatError(f"base32 decode failed: {exc}") from exc


def _canonical(payload: Dict) -> bytes:
    """Canonical JSON bytes — the exact bytes that get signed."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign(secret: bytes, payload_bytes: bytes) -> bytes:
    return hmac.new(secret, SIGNING_DOMAIN + b"." + payload_bytes, hashlib.sha256).digest()


def _key_id(payload_bytes: bytes) -> str:
    """Short, stable key id — safe to log, index, and revoke by."""
    return _b32encode(hashlib.sha256(payload_bytes).digest()[:6])


@dataclass
class IssuedKey:
    """The result of :func:`issue_key`."""

    key: str
    key_id: str
    payload: Dict


def generate_secret(nbytes: int = 32) -> bytes:
    """Generate a fresh vendor secret. Store it like a password."""
    return secrets.token_bytes(nbytes)


def issue_key(
    secret: bytes,
    email: str,
    plan: str = "trial",
    days: int = 30,
    sites: Optional[int] = None,
    features: Optional[Sequence[str]] = None,
    issued_at: Optional[datetime] = None,
) -> IssuedKey:
    """Issue a signed license key (vendor side).

    :param secret: vendor HMAC secret (bytes). Never ships to customers.
    :param email: customer identifier (email address).
    :param plan: plan name from :data:`PLANS`; unknown names raise ``ValueError``.
    :param days: validity window from issuance; ``0`` means no expiry.
    :param sites: override the plan's site allowance (``None`` = unlimited).
    :param features: override the plan's feature flags.
    :param issued_at: override issuance time (tests); defaults to now (UTC).
    """
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
        raise ValueError("secret must be bytes of at least 16 bytes")
    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}; known: {sorted(PLANS)}")
    email = email.strip().lower()
    if "@" not in email:
        raise ValueError(f"not an email address: {email!r}")
    if days < 0:
        raise ValueError("days must be >= 0")

    now = issued_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    plan_defaults = PLANS[plan]
    payload: Dict = {
        "v": KEY_VERSION,
        "email": email,
        "plan": plan,
        "sites": plan_defaults["sites"] if sites is None else sites,
        "feats": list(plan_defaults["features"]) if features is None else list(features),
        "iat": int(now.timestamp()),
        "exp": int(now.timestamp()) + days * 86400 if days else 0,
    }
    payload_bytes = _canonical(payload)
    payload["kid"] = _key_id(payload_bytes)
    # The kid is derived from the payload *without* itself, so re-canonicalize.
    payload_bytes = _canonical(payload)
    signature = _sign(bytes(secret), payload_bytes)
    key = f"{KEY_PREFIX}.{_b32encode(payload_bytes)}.{_b32encode(signature)}"
    return IssuedKey(key=key, key_id=payload["kid"], payload=payload)


def parse_key(key: str) -> Dict:
    """Parse a key string into its payload dict (no signature check).

    Raises :class:`KeyFormatError` on any structural problem. Use
    :func:`licensekit.validate.check_license` for full validation.
    """
    if not isinstance(key, str):
        raise KeyFormatError("key must be a string")
    parts = key.strip().split(".")
    if len(parts) != 3 or parts[0] != KEY_PREFIX:
        raise KeyFormatError(
            f"key must look like {KEY_PREFIX}.<payload>.<signature>"
        )
    payload_bytes = _b32decode(parts[1])
    signature = _b32decode(parts[2])
    if len(signature) != 32:
        raise KeyFormatError("signature must be 32 bytes (HMAC-SHA256)")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise KeyFormatError(f"payload is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise KeyFormatError("payload must be a JSON object")
    for required in ("v", "kid", "email", "plan", "feats", "iat", "exp"):
        if required not in payload:
            raise KeyFormatError(f"payload missing required field {required!r}")
    if payload["v"] != KEY_VERSION:
        raise KeyFormatError(
            f"unsupported key version {payload['v']!r} (this engine reads v{KEY_VERSION})"
        )
    # The kid must match the payload it claims to identify; otherwise the
    # payload was edited after the kid was computed.
    check = dict(payload)
    check.pop("kid", None)
    if _key_id(_canonical(check)) != payload["kid"]:
        raise KeyFormatError("key id does not match payload (payload was modified)")
    return {"payload": payload, "payload_bytes": payload_bytes, "signature": signature}


def verify_signature(secret: bytes, key: str) -> Dict:
    """Parse *and* verify a key's HMAC signature. Returns the payload dict.

    Raises :class:`KeyFormatError` for structural problems and
    :class:`LicenseError` for a bad signature. Prefer
    :func:`licensekit.validate.check_license`, which wraps this with expiry,
    revocation, and entitlement checks.
    """
    from .errors import LicenseError

    parsed = parse_key(key)
    expected = _sign(bytes(secret), parsed["payload_bytes"])
    if not hmac.compare_digest(expected, parsed["signature"]):
        raise LicenseError("bad signature: key was not issued with this secret")
    return parsed["payload"]


def fingerprint(key: str) -> str:
    """Short public identifier for a key — ``kid:<KEYID>``.

    Log this, never the full key. Returns ``"kid:invalid"`` for malformed keys
    so logging paths never raise.
    """
    try:
        return "kid:" + parse_key(key)["payload"]["kid"]
    except KeyFormatError:
        return "kid:invalid"
