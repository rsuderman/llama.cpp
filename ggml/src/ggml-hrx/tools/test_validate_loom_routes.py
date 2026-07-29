#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import validate_loom_routes as loom


TOOLS_DIR = Path(__file__).resolve().parent
CATALOG_ROOT = TOOLS_DIR.parent / "loom-catalog"
ROUTE_PATH = Path("routes/add_f32.json")


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_catalog(tmpdir):
    copied = Path(tmpdir) / "loom-catalog"
    copytree(CATALOG_ROOT, copied)
    return copied


def mutate_route(source_root, mutator):
    path = source_root / ROUTE_PATH
    route = read_json(path)
    mutator(route)
    write_json(path, route)


def expect_valid(source_root):
    loom.validate_catalog(source_root)


def expect_valid_mutation(name, mutator):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, mutator)
        loom.validate_catalog(source_root)


def expect_invalid(name, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def main():
    expect_valid(CATALOG_ROOT)
    expect_valid_mutation(
        "workgroups-dispatch",
        lambda route: route["invocation"].update({
            "dispatch": {"workgroups": ["derived.nelements"], "workgroup_size": [256, 1, 1]}
        }),
    )

    cases = [
        (
            "ambiguous-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"workgroups": ["derived.nelements"]}),
            "expects exactly one of work_items or workgroups",
        ),
        (
            "scalar-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"work_items": "derived.nelements"}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "too-many-dispatch-axes",
            lambda route: route["invocation"]["dispatch"].update({"work_items": [1, 1, 1, 1]}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "missing-config",
            lambda route: route.pop("config"),
            "expected object field config",
        ),
        (
            "duplicate-config-names",
            lambda route: route["config"]["bindings"][1].update({"name": "nelements"}),
            "duplicate config binding name nelements",
        ),
        (
            "unresolved-config-source",
            lambda route: route["config"]["bindings"][0].update({"source": "derived.missing"}),
            "derived value missing is not available",
        ),
        (
            "source-value-conflict",
            lambda route: route["config"]["bindings"][0].update({"value": 7}),
            "expected exactly one of source or value",
        ),
        (
            "forbidden-routing-field",
            lambda route: route.update({"routing": {}}),
            "unsupported field routing",
        ),
        (
            "forbidden-launch-field",
            lambda route: route.update({"launch": {}}),
            "unsupported field launch",
        ),
    ]

    for name, mutator, expected in cases:
        expect_invalid(name, mutator, expected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
