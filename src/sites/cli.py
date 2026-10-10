"""CLI: survey-sites validate | new | list | show | stories."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .packs import list_packs, load_pack
from .scaffold import TEMPLATES, scaffold, write_config
from .validate import ValidationError, assert_valid_file, validate_file


def cmd_validate(args: argparse.Namespace) -> int:
    errors = validate_file(args.config)
    if errors:
        for err in errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1
    print(f"OK: {args.config} is a valid site config (schema v1)")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    config = scaffold(args.site_id, args.name, template=args.template)
    path = write_config(config, args.out or f"{args.site_id}.yaml")
    print(f"wrote {path} (template: {args.template})")
    print("validate it with: survey-sites validate " + path)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    packs = list_packs(args.pack_dir)
    if not packs:
        print("no site packs found")
        return 1
    for pack in packs:
        status = "valid" if pack.valid else f"INVALID ({len(pack.errors)} errors)"
        print(f"{pack.site_id:20s} {pack.name:40s} [{status}]")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    pack = load_pack(args.directory)
    cfg = pack.config
    print(f"site_id : {pack.site_id}")
    print(f"name    : {cfg.get('name', '')}")
    print(f"indices : {', '.join(cfg.get('indices', []))}")
    zones = cfg.get("zones", [])
    print(f"zones   : {len(zones)}")
    for zone in zones:
        print(f"  - {zone.get('zone_id')} ({zone.get('kind', 'custom')}): {zone.get('name', '')}")
    transects = cfg.get("transects", [])
    if transects:
        print(f"transects: {len(transects)}")
        for tr in transects:
            print(f"  - {tr.get('transect_id')}: {tr.get('name', '')}")
    engines = cfg.get("engines", {})
    if engines:
        print(f"engines : {', '.join(sorted(engines))}")
    if pack.errors:
        print("INVALID:")
        for err in pack.errors:
            print(f"  {err}")
        return 1
    return 0


def cmd_stories(args: argparse.Namespace) -> int:
    packs = list_packs(args.pack_dir)
    for pack in packs:
        readme = os.path.join(pack.directory, "README.md")
        first_line = ""
        if os.path.exists(readme):
            with open(readme, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        first_line = line
                        break
        print(f"{pack.site_id}: {first_line}")
        demo = os.path.join(pack.directory, "demo.sh")
        if os.path.exists(demo):
            print(f"  demo: {demo}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey-sites",
        description="Canonical site-config schema and demo site pack for earthwatch-suite.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a site config file")
    v.add_argument("config", help="path to site.yaml / site.json")
    v.set_defaults(func=cmd_validate)

    n = sub.add_parser("new", help="scaffold a new site config from a template")
    n.add_argument("site_id", help="stable site id (lowercase, dashes/underscores)")
    n.add_argument("--name", default="", help="human-readable site name")
    n.add_argument("--template", choices=TEMPLATES, default="minimal")
    n.add_argument("--out", default="", help="output path (default: <site_id>.yaml)")
    n.set_defaults(func=cmd_new)

    l = sub.add_parser("list", help="list demo site packs")
    l.add_argument("--pack-dir", default=None, help="sites directory (default: shipped pack)")
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("show", help="inspect one site pack directory")
    s.add_argument("directory", help="path to a site directory")
    s.set_defaults(func=cmd_show)

    st = sub.add_parser("stories", help="list demo stories and their demo scripts")
    st.add_argument("--pack-dir", default=None)
    st.set_defaults(func=cmd_stories)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValidationError as exc:
        for err in exc.errors:
            print(f"INVALID: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
