"""survey-trends: weekly trending-sounds digest engine.

Reads TikTok's official Creative Center popular-music chart (keyless) and
renders a Markdown digest for Charlie's faceless data-viz reel pages.

Honest-access rule: if the source is unreachable or unparsable, the digest
says so and exits nonzero. Song titles, ranks, and counts are never
invented, and a stale cache is never presented as fresh.
"""

__version__ = "0.2.0"

from .models import (
    CredentialsMissingError,
    SongRank,
    SourceUnparsableError,
    SourceUnreachableError,
    TrendsError,
)

__all__ = [
    "__version__",
    "SongRank",
    "TrendsError",
    "SourceUnreachableError",
    "SourceUnparsableError",
    "CredentialsMissingError",
]
