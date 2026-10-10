"""The license gate: the ~3-line integration point for suite engines.

Intended usage inside survey-monitor / survey-alerts::

    from licensekit import license_gate

    @license_gate(feature="monitor", sites=1)
    def run_site(site_id, ...):
        ...

or as a context manager::

    from licensekit import licensed

    with licensed(feature="alerts-pdf") as lic:
        send_report(...)   # lic is the LicenseResult

The secret is resolved in this order: explicit ``secret=`` argument, then the
``SURVEY_LICENSE_SECRET`` environment variable, then ``secret_file=``. When no
secret is configured at all, validation cannot run — the gate then defers to
the :class:`LicensePolicy` for absent keys (trial defaults), which is the
documented fail-soft path for development. Engines that must keep working when
survey-license isn't installed at all should use the documented try/except
pattern (see README "Suite interop") instead of importing this module.
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from .errors import LicenseDenied
from .validate import DEFAULT_POLICY, LicensePolicy, LicenseResult, check_license

_LOG = logging.getLogger("licensekit.gate")

SECRET_ENV_VAR = "SURVEY_LICENSE_SECRET"


def resolve_secret(
    secret: Optional[bytes] = None, secret_file: Optional[str] = None
) -> Optional[bytes]:
    """Resolve the vendor secret: explicit arg → env var → secret file."""
    if secret is not None:
        return bytes(secret)
    env = os.environ.get(SECRET_ENV_VAR)
    if env:
        return env.encode("utf-8")
    if secret_file:
        with open(secret_file, "rb") as fh:
            return fh.read().strip()
    return None


def _check(
    key: Optional[str],
    feature: Optional[str],
    sites: int,
    secret: Optional[bytes],
    secret_file: Optional[str],
    policy: LicensePolicy,
    revocation_list,
    skew_seconds: int,
) -> LicenseResult:
    resolved = resolve_secret(secret, secret_file)
    if key and not resolved:
        # A key is present but we cannot verify it — that is a hard failure,
        # not a silent pass. Misconfiguration must be loud.
        raise LicenseDenied(
            "A license key was provided but no vendor secret is configured "
            f"(set {SECRET_ENV_VAR} or pass secret=/secret_file=)."
        )
    return check_license(
        key,
        resolved,
        required_feature=feature,
        sites=sites,
        revocation_list=revocation_list,
        skew_seconds=skew_seconds,
        policy=policy,
    )


def license_gate(
    feature: Optional[str] = None,
    sites: int = 1,
    *,
    secret: Optional[bytes] = None,
    secret_file: Optional[str] = None,
    key: Optional[str] = None,
    key_env_var: str = "SURVEY_LICENSE_KEY",
    policy: LicensePolicy = DEFAULT_POLICY,
    revocation_list=None,
    skew_seconds: int = 300,
) -> Callable:
    """Decorator enforcing a license check before the wrapped callable runs.

    The key is read from ``key=`` or the ``SURVEY_LICENSE_KEY`` env var.
    On denial, raises :class:`LicenseDenied` carrying the ``LicenseResult``.
    The full key is never logged — only its fingerprint.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            eff_key = key if key is not None else os.environ.get(key_env_var)
            result = _check(eff_key, feature, sites, secret, secret_file,
                            policy, revocation_list, skew_seconds)
            _LOG.info("license check for %s: %s (%s)",
                      getattr(func, "__name__", func), result.reason,
                      result.key_id or "no-key")
            if not result.valid:
                raise LicenseDenied(
                    f"License check failed for {getattr(func, '__name__', func)}: "
                    f"{result.reason} — {result.detail}",
                    result=result,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def licensed(
    feature: Optional[str] = None,
    sites: int = 1,
    *,
    secret: Optional[bytes] = None,
    secret_file: Optional[str] = None,
    key: Optional[str] = None,
    key_env_var: str = "SURVEY_LICENSE_KEY",
    policy: LicensePolicy = DEFAULT_POLICY,
    revocation_list=None,
    skew_seconds: int = 300,
) -> Iterator[LicenseResult]:
    """Context manager variant of :func:`license_gate`.

    Yields the :class:`LicenseResult`; raises :class:`LicenseDenied` when the
    check fails.
    """
    eff_key = key if key is not None else os.environ.get(key_env_var)
    result = _check(eff_key, feature, sites, secret, secret_file,
                    policy, revocation_list, skew_seconds)
    if not result.valid:
        raise LicenseDenied(
            f"License check failed: {result.reason} — {result.detail}",
            result=result,
        )
    yield result
