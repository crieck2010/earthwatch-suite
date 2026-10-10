"""survey-autopilot: automatic first-pass aesthetics engine.

Pure-logic, numpy-only engine for automatic first-pass aesthetics decisions in
the surveying / remote-sensing visualization stack. No survey-* imports, no UI
imports, duck-typed inputs — so it can never create an import cycle and can be
consumed as an optional peer by survey-viz, reel-studio, or survey-timescales.

Deterministic everywhere: scoring uses no randomness. Where randomness could
appear in downstream consumers (e.g. seeded strand placement), the seed is
owned by the caller, not by this package.
"""

__version__ = "0.2.0"

from .compose import DEFAULT_WEIGHTS, PRESET_FORMAT_TAGS, score_thumbnail, search_composition
from .density import strand_count_for_bbox
from .qa import DEFAULT_PLAUSIBLE, QAError, QAReport, qa_field, repair
from .salience import rank_label_positions, salience_map
from .scaling import robust_limits
from .stride import STRIDE_TABLE, recommend_stride

__all__ = [
    "__version__",
    # compose
    "DEFAULT_WEIGHTS",
    "PRESET_FORMAT_TAGS",
    "score_thumbnail",
    "search_composition",
    # scaling
    "robust_limits",
    # qa
    "QAReport",
    "QAError",
    "DEFAULT_PLAUSIBLE",
    "qa_field",
    "repair",
    # density
    "strand_count_for_bbox",
    # salience
    "salience_map",
    "rank_label_positions",
    # stride
    "STRIDE_TABLE",
    "recommend_stride",
]
