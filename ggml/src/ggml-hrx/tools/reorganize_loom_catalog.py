#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from utils import hrx_catalog_rebucket


def default_source_root():
    return Path(__file__).resolve().parents[1] / "loom-catalog"


def print_plan(plan):
    for issue in plan.issues:
        print(f"issue: {issue}")
    for move in plan.moves:
        print(f"move: {move.old_path} -> {move.new_path}")
        print(f"  {move.reason}")
    for replacement in plan.replacements:
        print(f"replace: {replacement.path}")
        print(f"  {replacement.old} -> {replacement.new}")
        print(f"  {replacement.reason}")
    for note in plan.notes:
        print(f"note: {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(default_source_root()))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--clean-empty-dirs", action="store_true")
    args = parser.parse_args()

    try:
        source_root = Path(args.source_root)
        plan = hrx_catalog_rebucket.collect_plan(source_root)
        has_changes = bool(plan.moves or plan.replacements)

        if args.explain:
            print_plan(plan)

        if args.check and (plan.issues or has_changes):
            if not args.explain:
                print_plan(plan)
            return 1

        if args.write:
            hrx_catalog_rebucket.apply_plan(source_root, plan, args.clean_empty_dirs)
            print(f"moved {len(plan.moves)} files and updated {len(plan.replacements)} references")
        elif not args.explain and has_changes:
            print_plan(plan)
    except (OSError, ValueError) as err:
        print(f"ValueError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
