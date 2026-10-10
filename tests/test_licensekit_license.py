"""Tests for survey-license. No network; fixed test secrets only."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from licensekit import (
    DEFAULT_POLICY,
    JsonFileKeyStore,
    LicenseDenied,
    LicensePolicy,
    check_license,
    fingerprint,
    issue_key,
    license_gate,
    licensed,
    parse_key,
)
from licensekit.errors import KeyFormatError, LicenseError
from licensekit.gate import resolve_secret
from licensekit.keys import PLANS, verify_signature
from licensekit.store import FileRevocationList

SECRET = b"test-vendor-secret-32-bytes-long!!"
OTHER_SECRET = b"a-different-test-secret-32bytes!"
NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


def issue(**kw):
    kw.setdefault("issued_at", NOW)
    return issue_key(SECRET, "customer@example.com", **kw)


# --- issuance & parsing -----------------------------------------------------

def test_issue_round_trip_all_plans():
    for plan in PLANS:
        issued = issue(plan=plan)
        assert issued.key.startswith("SL1.")
        payload = verify_signature(SECRET, issued.key)
        assert payload["plan"] == plan
        assert payload["email"] == "customer@example.com"
        assert payload["kid"] == issued.key_id


def test_issue_defaults_match_plan_catalogue():
    issued = issue(plan="site")
    assert issued.payload["sites"] == PLANS["site"]["sites"]
    assert issued.payload["feats"] == PLANS["site"]["features"]


def test_issue_overrides():
    issued = issue(plan="site", days=90, sites=7, features=["monitor", "alerts-email"])
    assert issued.payload["sites"] == 7
    assert issued.payload["feats"] == ["monitor", "alerts-email"]
    assert issued.payload["exp"] - issued.payload["iat"] == 90 * 86400


def test_issue_no_expiry():
    issued = issue(plan="unlimited", days=0)
    assert issued.payload["exp"] == 0


def test_issue_rejects_bad_inputs():
    with pytest.raises(ValueError):
        issue_key(b"short", "a@b.co")
    with pytest.raises(ValueError):
        issue_key(SECRET, "a@b.co", plan="nope")
    with pytest.raises(ValueError):
        issue_key(SECRET, "not-an-email")
    with pytest.raises(ValueError):
        issue_key(SECRET, "a@b.co", days=-1)


def test_key_is_copy_paste_safe():
    issued = issue()
    assert issued.key == issued.key.strip()
    assert " " not in issued.key and "\n" not in issued.key


def test_parse_key_structure():
    issued = issue()
    parsed = parse_key(issued.key)
    assert parsed["payload"]["email"] == "customer@example.com"
    assert len(parsed["signature"]) == 32


# --- tamper detection -------------------------------------------------------

def _flip(s: str, idx: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    c = s[idx]
    new_c = alphabet[(alphabet.index(c) + 1) % len(alphabet)] if c in alphabet else "A"
    return s[:idx] + new_c + s[idx + 1:]


def test_tampered_payload_fails_signature():
    issued = issue()
    prefix, payload_b32, sig_b32 = issued.key.split(".")
    tampered = f"{prefix}.{_flip(payload_b32, 5)}.{sig_b32}"
    with pytest.raises((KeyFormatError, LicenseError)):
        verify_signature(SECRET, tampered)


def test_tampered_signature_fails():
    issued = issue()
    prefix, payload_b32, sig_b32 = issued.key.split(".")
    tampered = f"{prefix}.{payload_b32}.{_flip(sig_b32, 5)}"
    with pytest.raises(LicenseError):
        verify_signature(SECRET, tampered)


def test_wrong_secret_fails():
    issued = issue()
    with pytest.raises(LicenseError):
        verify_signature(OTHER_SECRET, issued.key)


def test_malformed_keys_rejected():
    for bad in ["", "SL1", "SL1.abc", "XX1.abc.def", "SL1.!!!.???", 123]:
        with pytest.raises(KeyFormatError):
            parse_key(bad)


def test_kid_mismatch_detected():
    issued = issue()
    payload = dict(issued.payload)
    payload["sites"] = 9999  # attacker edits allowance, keeps kid
    from licensekit.keys import _b32encode, _canonical
    forged = f"SL1.{_b32encode(_canonical(payload))}.{issued.key.split('.')[2]}"
    with pytest.raises(KeyFormatError):
        parse_key(forged)


# --- validation: time -------------------------------------------------------

def test_valid_key_ok():
    issued = issue(plan="team", days=30)
    r = check_license(issued.key, SECRET, now=NOW)
    assert r.valid and r.reason == "ok"
    assert r.plan == "team" and r.email == "customer@example.com"


def test_expired_key():
    issued = issue(plan="site", days=30)
    r = check_license(issued.key, SECRET, now=NOW + timedelta(days=31))
    assert not r.valid and r.reason == "expired"


def test_no_expiry_key_never_expires():
    issued = issue(plan="unlimited", days=0)
    r = check_license(issued.key, SECRET, now=NOW + timedelta(days=3650))
    assert r.valid


def test_clock_skew_tolerance():
    issued = issue(plan="site", days=30)
    just_after = NOW + timedelta(days=30, seconds=60)  # within 300s skew
    r = check_license(issued.key, SECRET, now=just_after)
    assert r.valid, r.reason
    well_after = NOW + timedelta(days=30, seconds=301)
    r = check_license(issued.key, SECRET, now=well_after)
    assert not r.valid and r.reason == "expired"


def test_not_yet_valid():
    issued = issue(plan="site", days=30)
    r = check_license(issued.key, SECRET, now=NOW - timedelta(hours=1))
    assert not r.valid and r.reason == "not-yet-valid"


# --- validation: entitlements ------------------------------------------------

def test_feature_gating():
    trial = issue(plan="trial", days=30)
    r = check_license(trial.key, SECRET, required_feature="alerts-pdf", now=NOW)
    assert not r.valid and r.reason == "feature-denied"
    r = check_license(trial.key, SECRET, required_feature="monitor", now=NOW)
    assert r.valid


def test_site_allowance():
    site = issue(plan="site", days=30)  # 1 site
    assert check_license(site.key, SECRET, sites=1, now=NOW).valid
    r = check_license(site.key, SECRET, sites=2, now=NOW)
    assert not r.valid and r.reason == "too-many-sites"


def test_unlimited_sites():
    unlimited = issue(plan="unlimited", days=30)
    assert check_license(unlimited.key, SECRET, sites=10_000, now=NOW).valid


def test_result_helpers():
    team = issue(plan="team", days=30)
    r = check_license(team.key, SECRET, now=NOW)
    assert r.feature_enabled("alerts-email")
    assert not r.feature_enabled("nope")
    assert r.can_run_sites(10) and not r.can_run_sites(11)


# --- validation: absent keys & policy ----------------------------------------

def test_absent_key_trial_policy():
    r = check_license(None, SECRET, now=NOW)
    assert r.valid and r.reason == "trial-mode" and r.plan == "trial"


def test_absent_key_open_policy():
    r = check_license("", SECRET, policy=LicensePolicy(mode="open"), now=NOW)
    assert r.valid and r.reason == "no-enforcement"


def test_absent_key_strict_policy():
    r = check_license(None, SECRET, policy=LicensePolicy(mode="strict"), now=NOW)
    assert not r.valid and r.reason == "missing-key"


def test_bad_policy_mode_rejected():
    with pytest.raises(ValueError):
        LicensePolicy(mode="whatever")


# --- revocation ---------------------------------------------------------------

def test_revocation_blocks_key(tmp_path):
    issued = issue(plan="team", days=30)
    store = JsonFileKeyStore(str(tmp_path / "keys.json"))
    store.record_issuance("kid:" + issued.key_id, {"email": "customer@example.com"})
    assert check_license(issued.key, SECRET, revocation_list=store, now=NOW).valid
    assert store.revoke("kid:" + issued.key_id, reason="chargeback") is True
    r = check_license(issued.key, SECRET, revocation_list=store, now=NOW)
    assert not r.valid and r.reason == "revoked"


def test_revoke_unknown_key_id_still_blocks(tmp_path):
    store = JsonFileKeyStore(str(tmp_path / "keys.json"))
    assert store.revoke("kid:UNKNOWN123", reason="fraud") is False
    assert "kid:UNKNOWN123" in store  # blocklist semantics: unknown ids still block


def test_file_revocation_list(tmp_path):
    issued = issue(plan="team", days=30)
    path = tmp_path / "revoked.json"
    path.write_text(json.dumps({"revoked": ["kid:" + issued.key_id]}))
    r = check_license(issued.key, SECRET,
                      revocation_list=FileRevocationList(str(path)), now=NOW)
    assert not r.valid and r.reason == "revoked"
    assert check_license(issued.key, SECRET, now=NOW).valid  # no list → fine


def test_store_lists_issuances(tmp_path):
    store = JsonFileKeyStore(str(tmp_path / "keys.json"))
    a = issue(plan="trial")
    b = issue(plan="site")
    store.record_issuance("kid:" + a.key_id, {"email": "a@x.co", "issued_at": "2026-01-01"})
    store.record_issuance("kid:" + b.key_id, {"email": "b@x.co", "issued_at": "2026-02-01"})
    listed = store.list_keys()
    assert [k["key_id"] for k in listed] == ["kid:" + a.key_id, "kid:" + b.key_id]
    assert listed[0]["revoked"] is False


# --- fingerprinting / log hygiene ----------------------------------------------

def test_fingerprint_is_stable_and_short():
    issued = issue()
    fp = fingerprint(issued.key)
    assert fp == "kid:" + issued.key_id
    assert len(fp) < 20 and issued.key not in fp


def test_fingerprint_of_garbage_never_raises():
    assert fingerprint("garbage") == "kid:invalid"


def test_result_never_carries_full_key():
    issued = issue()
    r = check_license(issued.key, SECRET, now=NOW)
    assert r.key_id == "kid:" + issued.key_id
    assert issued.key not in repr(r) and issued.key not in str(r.__dict__)


# --- the gate -------------------------------------------------------------------

def test_gate_allows_with_valid_key(monkeypatch):
    issued = issue(plan="team", days=30)
    monkeypatch.setenv("SURVEY_LICENSE_KEY", issued.key)
    monkeypatch.setenv("SURVEY_LICENSE_SECRET", SECRET.decode())

    @license_gate(feature="alerts-email", sites=2)
    def protected():
        return "ran"

    assert protected() == "ran"


def test_gate_denies_and_carries_result(monkeypatch):
    issued = issue(plan="trial", days=30)  # trial lacks alerts-pdf
    monkeypatch.setenv("SURVEY_LICENSE_KEY", issued.key)
    monkeypatch.setenv("SURVEY_LICENSE_SECRET", SECRET.decode())

    @license_gate(feature="alerts-pdf")
    def protected():
        return "ran"

    with pytest.raises(LicenseDenied) as exc_info:
        protected()
    assert exc_info.value.result.reason == "feature-denied"


def test_gate_defers_to_trial_policy_without_key(monkeypatch):
    monkeypatch.delenv("SURVEY_LICENSE_KEY", raising=False)
    monkeypatch.delenv("SURVEY_LICENSE_SECRET", raising=False)

    @license_gate(feature="monitor")
    def protected():
        return "ran"

    assert protected() == "ran"  # trial-mode defaults


def test_gate_key_without_secret_is_loud(monkeypatch):
    issued = issue(plan="team", days=30)
    monkeypatch.setenv("SURVEY_LICENSE_KEY", issued.key)
    monkeypatch.delenv("SURVEY_LICENSE_SECRET", raising=False)

    @license_gate()
    def protected():
        return "ran"

    with pytest.raises(LicenseDenied):
        protected()


def test_licensed_context_manager_yields_result(monkeypatch):
    issued = issue(plan="site", days=30)
    monkeypatch.setenv("SURVEY_LICENSE_KEY", issued.key)
    monkeypatch.setenv("SURVEY_LICENSE_SECRET", SECRET.decode())
    with licensed(feature="monitor") as lic:
        assert lic.valid and lic.plan == "site"


def test_resolve_secret_precedence(tmp_path, monkeypatch):
    f = tmp_path / "secret.bin"
    f.write_bytes(b"file-secret-12345678")
    monkeypatch.setenv("SURVEY_LICENSE_SECRET", "env-secret-12345678")
    assert resolve_secret(b"explicit-secret-12") == b"explicit-secret-12"
    assert resolve_secret() == b"env-secret-12345678"
    monkeypatch.delenv("SURVEY_LICENSE_SECRET")
    assert resolve_secret(secret_file=str(f)) == b"file-secret-12345678"
    assert resolve_secret() is None


def test_expired_key_denied_through_gate(monkeypatch):
    issued = issue(plan="team", days=1)
    monkeypatch.setenv("SURVEY_LICENSE_KEY", issued.key)
    monkeypatch.setenv("SURVEY_LICENSE_SECRET", SECRET.decode())

    @license_gate()
    def protected():
        return "ran"

    # Freeze "now" via check_license's now param is not reachable through the
    # gate; instead issue an already-expired key relative to real now.
    old = issue_key(SECRET, "old@example.com", plan="team", days=1,
                    issued_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    monkeypatch.setenv("SURVEY_LICENSE_KEY", old.key)
    with pytest.raises(LicenseDenied) as exc_info:
        protected()
    assert exc_info.value.result.reason == "expired"


def test_empty_store_file_treated_as_fresh(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text("")  # e.g. pre-created by mktemp
    store = JsonFileKeyStore(str(path))
    assert store.list_keys() == []
    store.record_issuance("kid:ABC", {"email": "a@x.co"})
    assert len(JsonFileKeyStore(str(path)).list_keys()) == 1
