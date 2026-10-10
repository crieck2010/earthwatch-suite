"""survey-license: offline license-key engine for the earthwatch-suite monitoring product.

Pure stdlib. The vendor holds a secret; keys are HMAC-signed payloads that
engines validate locally with no phone-home requirement.
"""

from .errors import LicenseError, KeyFormatError, LicenseDenied
from .keys import (
    KEY_PREFIX,
    PLANS,
    IssuedKey,
    fingerprint,
    issue_key,
    parse_key,
)
from .validate import (
    DEFAULT_POLICY,
    LicensePolicy,
    LicenseResult,
    check_license,
)
from .store import KeyStore, JsonFileKeyStore, RevocationList, FileRevocationList
from .gate import license_gate, licensed

__version__ = "0.1.0"

__all__ = [
    "KEY_PREFIX",
    "PLANS",
    "DEFAULT_POLICY",
    "IssuedKey",
    "LicenseDenied",
    "LicenseError",
    "KeyFormatError",
    "KeyStore",
    "JsonFileKeyStore",
    "RevocationList",
    "FileRevocationList",
    "LicensePolicy",
    "LicenseResult",
    "check_license",
    "fingerprint",
    "issue_key",
    "license_gate",
    "licensed",
    "parse_key",
    "__version__",
]
