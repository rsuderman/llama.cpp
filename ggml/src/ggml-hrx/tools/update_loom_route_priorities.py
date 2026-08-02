#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

from utils import hrx_route_priority
from utils import hrx_route_schema as route_schema


PRIORITY_RE = re.compile(r'^(\s*"priority"\s*:\s*)-?\d+(,?\s*)$', re.MULTILINE)


def default_source_root():
    return Path(__file__).resolve().parents[1] / "loom-catalog"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def route_names(source_root):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    return route_schema.require_list(metadata, "routes", metadata_path)


def replace_priority_text(route_path, priority):
    text = route_path.read_text(encoding="utf-8")
    replacement_count = 0

    def replace(match):
        nonlocal replacement_count
        replacement_count += 1
        return f"{match.group(1)}{priority}{match.group(2)}"

    updated = PRIORITY_RE.sub(replace, text)
    if replacement_count != 1:
        raise ValueError(f"{route_path}: expected exactly one priority field, found {replacement_count}")
    if updated != text:
        route_path.write_text(updated, encoding="utf-8")
    return updated != text


def collect_priorities(source_root):
    priorities = []
    for route_name in route_names(source_root):
        if not isinstance(route_name, str) or not route_name:
            raise ValueError(f"{source_root / 'metadata.json'}: routes must contain non-empty strings")
        route_path = source_root / route_name
        route = read_json(route_path)
        current = route_schema.require_int(route, "priority", route_path)
        derived = hrx_route_priority.route_priority(route_name, route)
        priorities.append((route_name, route_path, current, derived))
    return priorities


def print_explanation(priorities):
    for route_name, _, current, derived in priorities:
        marker = "" if current == derived.priority else f" was {current}"
        print(f"{derived.priority} {route_name}{marker}")
        for reason in derived.reasons:
            print(f"  {reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(default_source_root()))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    try:
        source_root = Path(args.source_root)
        priorities = collect_priorities(source_root)
        changed = [(route_name, route_path, current, derived) for route_name, route_path, current, derived in priorities if current != derived.priority]

        if args.explain:
            print_explanation(priorities)

        if args.check and changed:
            for route_name, _, current, derived in changed:
                print(f"{route_name}: priority {current} should be {derived.priority}", file=sys.stderr)
            return 1

        if args.write:
            for _, route_path, _, derived in changed:
                replace_priority_text(route_path, derived.priority)
            print(f"updated {len(changed)} route priorities")
        elif not args.explain and changed:
            for route_name, _, current, derived in changed:
                print(f"{route_name}: priority {current} should be {derived.priority}")
    except (OSError, json.JSONDecodeError, ValueError) as err:
        print(f"ValueError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
