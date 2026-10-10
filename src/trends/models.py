"""Data models and error types for survey-trends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SongRank:
    """One ranked song on a trending-music chart."""

    rank: int                      # 1-based position on the chart
    title: str                     # song title, as reported by the source
    artist: str                    # artist name, as reported ("" when unknown)
    uses: Optional[int] = None     # videos/posts using the sound (None = not reported)
    trend: Optional[str] = None    # "up" | "down" | "new" | "same" | None
    source: str = "tiktok-creative-center"
    fetched_at: str = ""           # ISO-8601 UTC timestamp of the fetch

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "title": self.title,
            "artist": self.artist,
            "uses": self.uses,
            "trend": self.trend,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }


class TrendsError(Exception):
    """Base class for all survey-trends failures."""


class SourceUnreachableError(TrendsError):
    """The trend source could not be reached (network, HTTP, or auth failure)."""


class SourceUnparsableError(TrendsError):
    """The source responded but its payload shape was not recognized."""


class CredentialsMissingError(TrendsError):
    """A keyed source was requested without its credential.

    Raised *before* any request is made: nothing was fetched, nothing was
    sent, and no partial data is returned. The message always says exactly
    how to obtain the credential.
    """
