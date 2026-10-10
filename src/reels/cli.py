"""Thin CLI adapter for survey-reels. All logic lives in the engine modules."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_common(sp) -> None:
    sp.add_argument("--work-dir", default="reel_work",
                    help="working directory for stage artifacts (default: reel_work)")
    sp.add_argument("--cache-dir", default=None,
                    help="cache root (default: <work-dir>/.cache)")


def cmd_run(a: argparse.Namespace) -> int:
    from reels.recipe import ReelRecipe
    from reels.runner import run_recipe

    recipe = ReelRecipe.from_yaml(a.recipe)
    events = []

    def _progress(stage: str, done: int, total: int) -> None:
        events.append((stage, done, total))
        if a.verbose:
            print(f"[{stage}] {done}/{total}", flush=True)

    result = run_recipe(recipe, a.work_dir, cache_dir=a.cache_dir,
                        progress=_progress if a.verbose else None)
    print(json.dumps({
        "ok": result.ok,
        "video": result.video_path,
        "summary": result.summary_path,
        "stages_run": result.stages_run,
        "stages_skipped": result.stages_skipped,
        "errors": result.errors,
    }, indent=2))
    return 0 if result.ok else 1


def cmd_recipe_new(a: argparse.Namespace) -> int:
    from reels.recipe import example_recipe

    recipe = example_recipe()
    recipe.to_yaml(a.out)
    print(f"wrote {a.out}")
    return 0


def cmd_recipe_show(a: argparse.Namespace) -> int:
    from reels.recipe import ReelRecipe, recipe_hash

    recipe = ReelRecipe.from_yaml(a.recipe)
    print(json.dumps(recipe.to_dict(), indent=2))
    print(f"recipe_sha256: {recipe_hash(recipe)}")
    return 0


def cmd_from_site(a: argparse.Namespace) -> int:
    from reels.sites import recipe_from_site_file

    recipe = recipe_from_site_file(a.site)
    recipe.to_yaml(a.out)
    print(f"wrote {a.out} from {a.site}")
    return 0


def cmd_cache_clear(a: argparse.Namespace) -> int:
    from reels.cache import Cache

    cache = Cache(a.cache_dir or os.path.join(a.work_dir, ".cache"))
    removed = cache.clear()
    print(f"removed {removed} cached files")
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    """Fully offline demo: synthetic field -> frames -> video (needs peers)."""
    from reels.recipe import ReelRecipe
    from reels.runner import run_recipe

    recipe = ReelRecipe(
        name="demo-reel",
        source="synthetic",
        bbox=[-87.9, 41.6, -86.7, 42.9],
        n_frames=12,
        n_particles=400,
        flow_style="reel-dark",
        animate_preset="preview",
        fps=15.0,
        title="survey-reels demo",
        subtitle="synthetic currents · offline",
        output="demo-reel.mp4",
    )
    result = run_recipe(recipe, a.work_dir, cache_dir=a.cache_dir)
    print(json.dumps({
        "ok": result.ok,
        "video": result.video_path,
        "summary": result.summary_path,
        "errors": result.errors,
    }, indent=2))
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reels",
                                description="Orchestrate current/temperature reels")
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run a recipe YAML end to end")
    s.add_argument("recipe", help="recipe YAML file")
    _add_common(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("recipe-new", help="write an example recipe YAML")
    s.add_argument("--out", default="reel.yaml")
    s.set_defaults(func=cmd_recipe_new)

    s = sub.add_parser("recipe-show", help="validate and print a recipe YAML")
    s.add_argument("recipe")
    s.set_defaults(func=cmd_recipe_show)

    s = sub.add_parser("from-site",
                       help="build a recipe YAML from a survey-sites site.yaml")
    s.add_argument("site", help="path to site.yaml")
    s.add_argument("--out", default="reel.yaml")
    s.set_defaults(func=cmd_from_site)

    s = sub.add_parser("demo", help="offline end-to-end demo (needs peer packages)")
    _add_common(s)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("cache-clear", help="clear the content-addressed cache")
    _add_common(s)
    s.set_defaults(func=cmd_cache_clear)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI surface: print and exit 1
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
