#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import generate_loom_catalog as catalog
import generate_loom_route_impl as route_impl
import validate_loom_routes as loom


TOOLS_DIR = Path(__file__).resolve().parent
CATALOG_ROOT = TOOLS_DIR.parent / "loom-catalog"
METADATA_PATH = Path("metadata.json")
ROUTE_PATH = Path("routes/generic/add/f32/contiguous.json")
SUM_ROWS_ROUTE_PATH = Path("routes/generic/sum_rows/f32/contiguous_4d.json")
FUSION_ROUTE_PATH = Path("routes/generic/rms_norm_mul/f32/contiguous_4d.json")
TEST_TARGET_A = "__test_target_a"
TEST_TARGET_B = "__test_target_b"
TEST_TARGET_MISSING = "__test_missing_target"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_route_and_definition(source_root, route_path):
    route_full_path = source_root / route_path
    route = read_json(route_full_path)
    definition = read_json((route_full_path.parent / route["definition"]).resolve())
    return route, definition


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def copy_catalog(tmpdir):
    copied = Path(tmpdir) / "loom-catalog"
    copytree(CATALOG_ROOT, copied)
    mutate_metadata(copied, lambda metadata: metadata.update({"routes": [str(ROUTE_PATH)]}))
    return copied


def mutate_route(source_root, mutator):
    mutate_route_at(source_root, ROUTE_PATH, mutator)


def mutate_route_at(source_root, route_path, mutator):
    path = source_root / route_path
    route = read_json(path)
    mutator(route)
    write_json(path, route)


def mutate_metadata(source_root, mutator):
    path = source_root / METADATA_PATH
    metadata = read_json(path)
    mutator(metadata)
    write_json(path, metadata)


def set_metadata_targets(source_root, targets):
    mutate_metadata(source_root, lambda metadata: metadata.update({"targets": targets}))


def set_route_architectures(source_root, architectures):
    mutate_route(source_root, lambda route: route.update({"architectures": architectures}))


def set_targets_and_architectures(source_root, targets, architectures):
    set_metadata_targets(source_root, targets)
    set_route_architectures(source_root, architectures)


def expect_valid(source_root):
    loom.validate_catalog(source_root)


def expect_valid_mutation(name, mutator):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, mutator)
        loom.validate_catalog(source_root)


def expect_valid_catalog_mutation(name, mutator):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
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


def expect_invalid_catalog_mutation(name, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def expect_invalid_full_catalog_mutation(name, route_path, mutator, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = Path(tmpdir) / "loom-catalog"
        copytree(CATALOG_ROOT, source_root)
        mutate_route_at(source_root, route_path, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: validator accepted invalid catalog")


def expect_generation_valid(name, mutator, targets, expected_entries):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        loom.validate_catalog(source_root)
        entries = catalog.build_entries(source_root, targets)
        if len(entries) != len(expected_entries):
            raise AssertionError(f"{name}: expected {len(expected_entries)} entries, got {len(entries)}")
        actual_entries = [{"id": entry["id"], "target": entry["target"]} for entry in entries]
        if actual_entries != expected_entries:
            raise AssertionError(f"{name}: expected generated entries {expected_entries}, got {actual_entries}")


def expect_generation_invalid(name, mutator, targets, expected):
    with tempfile.TemporaryDirectory(prefix=f"{name}-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutator(source_root)
        loom.validate_catalog(source_root)
        try:
            catalog.build_entries(source_root, targets)
        except ValueError as err:
            message = str(err)
            if expected in message:
                return
            raise AssertionError(f"{name}: expected error containing {expected!r}, got {message!r}") from err
        raise AssertionError(f"{name}: generator accepted invalid catalog")


def main():
    expect_valid(CATALOG_ROOT)
    sum_rows_route, sum_rows_definition = read_route_and_definition(CATALOG_ROOT, SUM_ROWS_ROUTE_PATH)
    sum_rows_impl = route_impl.generate_route_impl(str(SUM_ROWS_ROUTE_PATH), sum_rows_route, sum_rows_definition)
    if "shape_dst_d0 != shape_src0_d0" in sum_rows_impl:
        raise AssertionError("v1 shape captures must not imply cross-tensor equality")

    fusion_route, fusion_definition = read_route_and_definition(CATALOG_ROOT, FUSION_ROUTE_PATH)
    fusion_impl = route_impl.generate_route_impl(str(FUSION_ROUTE_PATH), fusion_route, fusion_definition)
    if "shape_rms_out_d0 != shape_x_d0" not in fusion_impl:
        raise AssertionError("v2 structural shape declarations must enforce cross-tensor equality")

    expect_valid_mutation(
        "workgroups-dispatch",
        lambda route: route["invocation"].update({
            "dispatch": {"workgroups": ["derived.total_size"], "workgroup_size": [256, 1, 1]}
        }),
    )

    cases = [
        (
            "missing-architectures",
            lambda route: route.pop("architectures"),
            "expected array field architectures",
        ),
        (
            "empty-architectures",
            lambda route: route.update({"architectures": []}),
            "architectures must not be empty",
        ),
        (
            "scalar-architectures",
            lambda route: route.update({"architectures": "gfx1100"}),
            "expected array field architectures",
        ),
        (
            "empty-architecture",
            lambda route: route.update({"architectures": [""]}),
            "expected non-empty string",
        ),
        (
            "non-string-architecture",
            lambda route: route.update({"architectures": [7]}),
            "expected non-empty string",
        ),
        (
            "duplicate-architectures",
            lambda route: route.update({"architectures": ["gfx1100", "gfx1100"]}),
            "duplicate architecture gfx1100",
        ),
        (
            "mixed-any-architecture",
            lambda route: route.update({"architectures": ["*", "gfx1100"]}),
            "architecture * cannot be combined with explicit architectures",
        ),
        (
            "unknown-architecture",
            lambda route: route.update({"architectures": [TEST_TARGET_MISSING]}),
            f"architecture {TEST_TARGET_MISSING} is not listed in metadata targets",
        ),
        (
            "ambiguous-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"workgroups": ["derived.total_size"]}),
            "expects exactly one of work_items or workgroups",
        ),
        (
            "scalar-dispatch",
            lambda route: route["invocation"]["dispatch"].update({"work_items": "derived.total_size"}),
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
            lambda route: route["config"]["bindings"][1].update({"name": "shape_pointwise_total_size"}),
            "duplicate config binding name shape_pointwise_total_size",
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

    expect_invalid_full_catalog_mutation(
        "fusion-unknown-transient",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][2].update({"transients": ["missing"]}),
        "tensor missing is not declared in tensors",
    )
    expect_invalid_full_catalog_mutation(
        "fusion-no-overlap-arity",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][3].update({"no_overlap": ["x"]}),
        "expected two tensor names",
    )

    expect_invalid_catalog_mutation(
        "missing-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.pop("targets")),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "scalar-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.update({"targets": TEST_TARGET_A})),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "empty-metadata-targets",
        lambda source_root: mutate_metadata(source_root, lambda metadata: metadata.update({"targets": []})),
        "expected non-empty list field targets",
    )
    expect_invalid_catalog_mutation(
        "non-string-metadata-target",
        lambda source_root: set_metadata_targets(source_root, ["gfx1100", 7]),
        "targets must contain non-empty strings",
    )
    expect_invalid_catalog_mutation(
        "duplicate-metadata-targets",
        lambda source_root: set_metadata_targets(source_root, [TEST_TARGET_A, TEST_TARGET_A]),
        f"duplicate target {TEST_TARGET_A}",
    )
    expect_invalid_catalog_mutation(
        "empty-metadata-target",
        lambda source_root: set_metadata_targets(source_root, ["gfx1100", ""]),
        "targets must contain non-empty strings",
    )

    expect_invalid_catalog_mutation(
        "route-architecture-subset",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            [TEST_TARGET_A],
        ),
        f"metadata target {TEST_TARGET_B} is not covered by any route architecture",
    )
    expect_valid_catalog_mutation(
        "route-architecture-any",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            ["*"],
        ),
    )
    expect_generation_valid(
        "generate-compatible-subset",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            [TEST_TARGET_A, TEST_TARGET_B],
        ),
        [TEST_TARGET_A],
        [{"id": "add_f32_contiguous", "target": TEST_TARGET_A}],
    )
    expect_generation_valid(
        "generate-any-architecture",
        lambda source_root: set_targets_and_architectures(
            source_root,
            [TEST_TARGET_A, TEST_TARGET_B],
            ["*"],
        ),
        [TEST_TARGET_A, TEST_TARGET_B],
        [
            {"id": "add_f32_contiguous", "target": TEST_TARGET_A},
            {"id": "add_f32_contiguous", "target": TEST_TARGET_B},
        ],
    )
    expect_generation_invalid(
        "generate-unknown-target",
        lambda source_root: set_targets_and_architectures(source_root, [TEST_TARGET_A], [TEST_TARGET_A]),
        [TEST_TARGET_B],
        f"selected target {TEST_TARGET_B} is not listed in metadata targets",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
