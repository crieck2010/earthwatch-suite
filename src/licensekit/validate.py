"""Offline license validation: signature, expiry, revocation, entitlements.

The validation pipeline, in order:

1. **Presence** — no key at all → the license *policy* decides (trial-mode
   defaults, open, or strict). The mechanism is fixed; the policy is
   configurable (:class:`LicensePolicy`).
2. **Format** — structural parse (:func:`licensekit.keys.parse_key`).
3. **Signature** — HMAC check against the vendor secret.
4. **Time** — expiry enforced with clock-skew tolerance; not-yet-valid
   issuance rejected.
5. **Revocation** — key id checked against a revocation list.
6. **Entitlements** — required feature flag present; requested site count
   within the allowance.

Every outcome is a :class:`LicenseResult` with a machine-readable ``reason``
(``"ok"``, ``"expired"``, ``"revoked"``, ``"bad-signature"``, ...), so engines
can branch on it without string-matching messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .errors import KeyFormatError, LicenseError
from .keys import KNOWN_FEATURES, PLANS, fingerprint, verify_signature

#: Machine-readable outcome codes returned in ``LicenseResult.reason``.
REASONS = (
    "ok",
    "trial-mode",       # no key; policy granted trial defaults
    "no-enforcement",   # no key; policy is open
    "missing-key",      # no key; policy is strict
    "bad-format",
    "bad-signature",
    "not-yet-valid",
    "expired",
    "revoked",
    "feature-denied",
    "too-many-sites",
)


@dataclass(frozen=True)
class LicensePolicy:
    """What to do when no license key is present.

    The *mechanism* (how keys are checked) is fixed; the *policy* (what an
    absent key means) is a deployment choice:

    * ``"trial"`` (default) — grant time-boxed trial defaults so a new user
      can evaluate the product before buying.
    * ``"open"`` — no enforcement at all (dev/test, or a product that gates
      only at the payment layer).
    * ``"strict"`` — no key means no access.
    """

    mode: str = "trial"  # "trial" | "open" | "strict"
    trial_days: int = 14
    trial_sites: int = 1
    trial_features: Sequence[str] = ("monitor",)

    def __post_init__(self):
        if self.mode not in ("trial", "open", "strict"):
            raise ValueError(f"unknown policy mode {self.mode!r}")


DEFAULT_POLICY = LicensePolicy()


@dataclass(frozen=True)
class LicenseResult:
    """The outcome of :func:`check_license`."""

    valid: bool
    reason: str
    plan: str
    email: Optional[str]
    features: Sequence[str]
    sites: Optional[int]  # None = unlimited
    expires_at: Optional[datetime]
    key_id: Optional[str]  # fingerprint form, "kid:XXXX" — never the raw key
    detail: str = ""

    def feature_enabled(self, name: str) -> bool:
        return self.valid and name in self.features

    def can_run_sites(self, n: int) -> bool:
        if not self.valid:
            return False
        return self.sites is None or n <= self.sites


def _as_utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def check_license(
    key: Optional[str],
    secret: Optional[bytes] = None,
    required_feature: Optional[str] = None,
    sites: int = 1,
    *,
    now: Optional[datetime] = None,
    revocation_list=None,
    skew_seconds: int = 300,
    policy: LicensePolicy = DEFAULT_POLICY,
) -> LicenseResult:
    """Validate a license key and its entitlements (offline, no network).

    :param key: the license key string, or ``None``/``""`` if absent.
    :param secret: vendor HMAC secret (bytes). Required when a key is given.
    :param required_feature: feature flag the caller needs (e.g.
        ``"alerts-pdf"``); denied when absent from the key.
    :param sites: how many sites this run will touch; denied when above the
        key's allowance (``None`` allowance = unlimited).
    :param now: override "now" (tests); defaults to UTC now.
    :param revocation_list: object supporting ``kid in revocation_list``
        (see :mod:`licensekit.store`); ``None`` disables revocation checks.
    :param skew_seconds: clock-skew tolerance applied to expiry/issuance.
    :param policy: :class:`LicensePolicy` governing absent keys.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # 1. Presence — policy decides.
    if not key or not key.strip():
        if policy.mode == "trial":
            return LicenseResult(
                valid=True,
                reason="trial-mode",
                plan="trial",
                email=None,
                features=tuple(policy.trial_features),
                sites=policy.trial_sites,
                expires_at=None,  # trial window starts at first gated run; engines track it
                key_id=None,
                detail=f"No license key; trial policy grants {policy.trial_days} days, "
                f"{policy.trial_sites} site(s).",
            )
        if policy.mode == "open":
            return LicenseResult(
                valid=True, reason="no-enforcement", plan="open", email=None,
                features=tuple(KNOWN_FEATURES), sites=None, expires_at=None,
                key_id=None, detail="No license key; policy is open (no enforcement).",
            )
        return LicenseResult(
            valid=False, reason="missing-key", plan="none", email=None,
            features=(), sites=0, expires_at=None, key_id=None,
            detail="No license key and policy is strict.",
        )

    # 2+3. Format and signature.
    try:
        payload = verify_signature(bytes(secret), key)
    except KeyFormatError as exc:
        return LicenseResult(
            valid=False, reason="bad-format", plan="none", email=None,
            features=(), sites=0, expires_at=None,
            key_id=fingerprint(key), detail=str(exc),
        )
    except LicenseError as exc:
        return LicenseResult(
            valid=False, reason="bad-signature", plan="none", email=None,
            features=(), sites=0, expires_at=None,
            key_id=fingerprint(key), detail=str(exc),
        )
    except (TypeError, ValueError) as exc:  # secret missing/malformed
        raise LicenseError(f"cannot validate without a proper secret: {exc}") from exc

    kid = "kid:" + payload["kid"]
    exp_ts = int(payload["exp"])
    iat_ts = int(payload["iat"])
    expires_at = _as_utc(exp_ts) if exp_ts else None

    base = dict(
        plan=str(payload["plan"]),
        email=str(payload["email"]),
        features=tuple(payload["feats"]),
        sites=payload["sites"],
        expires_at=expires_at,
        key_id=kid,
    )

    def denied(reason: str, detail: str) -> LicenseResult:
        return LicenseResult(valid=False, reason=reason, detail=detail, **base)

    # 4. Time.
    now_ts = now.timestamp()
    if iat_ts > now_ts + skew_seconds:
        return denied("not-yet-valid",
                      f"Key issued in the future (iat={_as_utc(iat_ts).isoformat()}).")
    if exp_ts and now_ts > exp_ts + skew_seconds:
        return denied("expired",
                      f"Key expired {expires_at.isoformat()} "
                      f"(skew tolerance {skew_seconds}s).")

    # 5. Revocation.
    if revocation_list is not None and kid in revocation_list:
        return denied("revoked", f"Key {kid} is on the revocation list.")

    # 6. Entitlements.
    if required_feature is not None and required_feature not in payload["feats"]:
        return denied("feature-denied",
                      f"Plan {payload['plan']!r} does not include "
                      f"feature {required_feature!r}.")
    allowance = payload["sites"]
    if allowance is not None and sites > int(allowance):
        return denied("too-many-sites",
                      f"Run needs {sites} site(s); key allows {allowance}.")

    return LicenseResult(valid=True, reason="ok",
                         detail="Signature valid; within expiry and entitlements.",
                         **base)
