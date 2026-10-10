"""survey-derive: derived-products engine for the survey/reel stack.

Pure computation, no network, no UI, no matplotlib. Sits between data
fetching (survey-currents) and rendering (survey-viz): gridded
time-series fields go in, derived fields come out in the same
plain-dict form the renderer consumes.

NaN means missing throughout — never zero-filled, never silently
dropped from counts.
"""

from .climatology import Climatology, climatology
from .products import (
    anomaly,
    monthly_means,
    percent_of_normal,
    standardized_anomaly,
    suggest_symmetric_limits,
    trend,
)

__version__ = "0.1.0"

__all__ = [
    "Climatology",
    "__version__",
    "anomaly",
    "climatology",
    "monthly_means",
    "percent_of_normal",
    "standardized_anomaly",
    "suggest_symmetric_limits",
    "trend",
]
