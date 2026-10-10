"""Error types for survey-license."""

from __future__ import annotations


class LicenseError(Exception):
    """Base class for all survey-license errors."""


class KeyFormatError(LicenseError):
    """A license key could not be parsed (bad prefix, bad encoding, bad shape)."""


class LicenseDenied(LicenseError):
    """Raised by the license gate when an action is not permitted.

    Carries the :class:`licensekit.validate.LicenseResult` that caused the
    denial in :attr:`result`, so callers can report *why* (expired, revoked,
    feature-denied, ...) without re-checking.
    """

    def __init__(self, message: str, result=None):
        super().__init__(message)
        self.result = result
