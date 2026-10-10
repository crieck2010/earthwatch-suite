"""survey-license CLI: issue, validate, revoke, entitlements, fingerprint.

Vendor secret resolution: ``--secret-file`` beats ``--secret-env`` (default
``SURVEY_LICENSE_SECRET``). The secret is never printed or logged.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone

from . import __version__
from .errors import LicenseError
from .gate import SECRET_ENV_VAR, resolve_secret
from .keys import PLANS, fingerprint, generate_secret, issue_key
from .store import FileRevocationList, JsonFileKeyStore
from .validate import LicensePolicy, check_license


def _get_secret(args) -> bytes:
    secret = resolve_secret(secret_file=getattr(args, "secret_file", None))
    if secret is None and getattr(args, "secret_env", None):
        env = os.environ.get(args.secret_env)
        if env:
            secret = env.encode("utf-8")
    if secret is None:
        # Last resort: prompt (getpass — never echoed). Only for interactive use.
        try:
            typed = getpass.getpass("Vendor secret: ")
        except (EOFError, KeyboardInterrupt):
            typed = ""
        if typed:
            secret = typed.encode("utf-8")
    if secret is None:
        raise LicenseError(
            f"No vendor secret: set {SECRET_ENV_VAR}, pass --secret-file, or use --secret-env."
        )
    return secret


def _add_secret_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--secret-file", default=None,
                   help="Path to a file containing the vendor secret.")
    p.add_argument("--secret-env", default=SECRET_ENV_VAR,
                   help=f"Env var holding the vendor secret (default: {SECRET_ENV_VAR}).")


def _result_json(result) -> dict:
    return {
        "valid": result.valid,
        "reason": result.reason,
        "plan": result.plan,
        "email": result.email,
        "features": list(result.features),
        "sites": result.sites,
        "expires_at": result.expires_at.isoformat() if result.expires_at else None,
        "key_id": result.key_id,
        "detail": result.detail,
    }


def cmd_issue(args) -> int:
    secret = _get_secret(args)
    store = JsonFileKeyStore(args.store) if args.store else None
    issued = issue_key(
        secret,
        args.email,
        plan=args.plan,
        days=args.days,
        sites=args.sites,
        features=args.features.split(",") if args.features else None,
    )
    if store:
        store.record_issuance(issued.key_id, {
            "email": issued.payload["email"],
            "plan": issued.payload["plan"],
            "sites": issued.payload["sites"],
            "features": issued.payload["feats"],
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.fromtimestamp(issued.payload["exp"], tz=timezone.utc).isoformat()
                           if issued.payload["exp"] else None),
        })
        print(f"recorded in {args.store}", file=sys.stderr)
    # The key is printed once, to stdout — the vendor's job is to deliver it.
    print(issued.key)
    print(f"key_id: kid:{issued.key_id}", file=sys.stderr)
    return 0


def cmd_validate(args) -> int:
    secret = _get_secret(args)
    revocation = FileRevocationList(args.revocation_list) if args.revocation_list else None
    result = check_license(
        args.key,
        secret,
        required_feature=args.feature,
        sites=args.sites,
        revocation_list=revocation,
    )
    print(json.dumps(_result_json(result), indent=2))
    return 0 if result.valid else 2


def cmd_revoke(args) -> int:
    store = JsonFileKeyStore(args.store)
    known = store.revoke(args.key_id, reason=args.reason or "")
    print(json.dumps({"key_id": args.key_id, "revoked": True,
                      "previously_known": known}, indent=2))
    return 0


def cmd_entitlements(args) -> int:
    secret = _get_secret(args)
    result = check_license(args.key, secret)
    print(json.dumps(_result_json(result), indent=2))
    return 0 if result.valid else 2


def cmd_fingerprint(args) -> int:
    print(fingerprint(args.key))
    return 0


def cmd_gen_secret(_args) -> int:
    # Prints a fresh secret as hex. Store it like a password; it is never
    # stored by this tool.
    print(generate_secret().hex())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-license",
        description="Offline license-key engine for the earthwatch-suite monitoring product.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("issue", help="Issue a signed license key (vendor side).")
    pi.add_argument("--email", required=True, help="Customer email address.")
    pi.add_argument("--plan", default="trial", choices=sorted(PLANS),
                    help="Plan template (default: trial).")
    pi.add_argument("--days", type=int, default=30,
                    help="Validity in days from now; 0 = no expiry (default: 30).")
    pi.add_argument("--sites", type=int, default=None,
                    help="Override the plan's site allowance (omit = plan default).")
    pi.add_argument("--features", default=None,
                    help="Comma-separated feature override (omit = plan default).")
    pi.add_argument("--store", default=None,
                    help="Vendor key-store JSON file to record issuance in.")
    _add_secret_args(pi)
    pi.set_defaults(func=cmd_issue)

    pv = sub.add_parser("validate", help="Validate a key and its entitlements.")
    pv.add_argument("key", help="The license key string.")
    pv.add_argument("--feature", default=None, help="Required feature flag.")
    pv.add_argument("--sites", type=int, default=1, help="Sites this run touches.")
    pv.add_argument("--revocation-list", default=None,
                    help="JSON revocation-list file to check against.")
    _add_secret_args(pv)
    pv.set_defaults(func=cmd_validate)

    pr = sub.add_parser("revoke", help="Revoke a key id in the vendor store.")
    pr.add_argument("key_id", help="Key id (kid:XXXX form or bare).")
    pr.add_argument("--store", required=True, help="Vendor key-store JSON file.")
    pr.add_argument("--reason", default="", help="Reason for revocation.")
    pr.set_defaults(func=cmd_revoke)

    pe = sub.add_parser("entitlements", help="Show what a key entitles its holder to.")
    pe.add_argument("key", help="The license key string.")
    _add_secret_args(pe)
    pe.set_defaults(func=cmd_entitlements)

    pf = sub.add_parser("fingerprint", help="Print the log-safe fingerprint of a key.")
    pf.add_argument("key", help="The license key string.")
    pf.set_defaults(func=cmd_fingerprint)

    pg = sub.add_parser("gen-secret", help="Generate a fresh vendor secret (hex).")
    pg.set_defaults(func=cmd_gen_secret)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LicenseError as exc:
        print(f"survey-license: error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
