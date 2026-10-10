"""survey-gazetteer: offline place-label engine for map reels.

Bundles Natural Earth 1:10m populated places (~7k records) and selects
ranked label candidates for a region bbox, in exactly the format
survey-aesthetics' ``place_labels`` consumes. Stdlib only, no network.
"""

from gazetteer.api import labels_for_bbox, search
from gazetteer.places import dataset_info, load_places
from gazetteer.scoring import DEFAULT_KINDS, kind_of, priority_of

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "labels_for_bbox",
    "search",
    "load_places",
    "dataset_info",
    "DEFAULT_KINDS",
    "kind_of",
    "priority_of",
]
