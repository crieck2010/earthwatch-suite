"""survey-reels: orchestrate mapped.earth-style current/temperature reels.

Thin orchestration engine — all heavy lifting lives in the optional peer
packages ``survey-currents`` (fetch), ``survey-flow`` (render) and
``survey-animate`` (encode). This package only wires them together with a
recipe schema, resumable runs, caching, and site-config integration.
"""

__version__ = "0.1.0"

from reels.recipe import ReelRecipe, recipe_hash
from reels.runner import run_recipe, RunResult, RunState

__all__ = ["__version__", "ReelRecipe", "recipe_hash", "run_recipe", "RunResult", "RunState"]
