"""survey-sites: canonical site-config schema and demo site pack.

The site config is the cross-suite contract that survey-monitor,
survey-alerts, and every earthwatch-suite analytics engine consume.
This package validates configs, scaffolds new ones, and ships the demo
site pack.
"""

from .validate import ValidationError, validate_config, validate_file
from .packs import SitePack, load_pack, list_packs

__version__ = "0.1.0"
__all__ = [
    "ValidationError",
    "validate_config",
    "validate_file",
    "SitePack",
    "load_pack",
    "list_packs",
]
