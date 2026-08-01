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
    definition = read_json((route_full_path.parent / route["dispatches"][0]["definition"]).resolve())
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


def mutate_definition(source_root, mutator):
    route_path = source_root / ROUTE_PATH
    route = read_json(route_path)
    definition_path = (route_path.parent / route["dispatches"][0]["definition"]).resolve()
    definition = read_json(definition_path)
    mutator(definition)
    write_json(definition_path, definition)


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


def add_dependency_source(source_root):
    source_name = "sources/generic/add/f32/helper.loom"
    source_path = source_root / source_name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("kernel.func @helper() { }\n", encoding="utf-8")
    return source_name


def set_definition_dependencies(source_root, dependencies):
    mutate_definition(source_root, lambda definition: definition.update({"dependencies": dependencies}))


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


def expect_dependency_generation_valid():
    with tempfile.TemporaryDirectory(prefix="generate-dependencies-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        dependency_source = add_dependency_source(source_root)
        set_definition_dependencies(
            source_root,
            [{"source": dependency_source, "source_format": "loom-text"}],
        )
        loom.validate_catalog(source_root)
        entries = catalog.build_entries(source_root, ["gfx1100"])
        if len(entries) != 1:
            raise AssertionError(f"generate-dependencies: expected 1 entry, got {len(entries)}")
        dependencies = entries[0]["dependencies"]
        if len(dependencies) != 1:
            raise AssertionError(f"generate-dependencies: expected 1 dependency, got {len(dependencies)}")
        if dependencies[0]["source_name"] != dependency_source:
            raise AssertionError("generate-dependencies: dependency source name was not preserved")
        if dependencies[0]["source_format"] != "loom-text":
            raise AssertionError("generate-dependencies: dependency source format was not preserved")


def expect_index_scalar_valid():
    with tempfile.TemporaryDirectory(prefix="index-scalar-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def add_index_parameter(definition):
            definition["abi"].update({"parameter_count": 4, "constant_byte_length": 4})
            definition["parameters"] = [{"name": "tile_count", "type": "index"}]

        def add_index_scalar(route):
            route["dispatches"][0]["scalars"] = [
                {"name": "tile_count", "type": "index", "position": 0, "source": "shape.dst.d0"}
            ]

        mutate_definition(source_root, add_index_parameter)
        mutate_route(source_root, add_index_scalar)
        loom.validate_catalog(source_root)

        route, definition = read_route_and_definition(source_root, ROUTE_PATH)
        impl = route_impl.generate_route_impl(str(ROUTE_PATH), route, definition)
        if "const int32_t constant_tile_count = static_cast<int32_t>(shape_dst_d0);" not in impl:
            raise AssertionError("index-scalar: expected index scalar to pack as int32_t")


def make_multi_step_add_route(route):
    base_dispatch = route["dispatches"][0]
    definition = base_dispatch["definition"]
    config = base_dispatch["config"]
    route["derived"]["total_bytes"] = {
        "type": "i64",
        "product": ["derived.total_size", 4],
    }
    route["transient_buffers"] = {
        "tmp": {"size": "derived.total_bytes"},
    }
    first_dispatch = json.loads(json.dumps(base_dispatch))
    first_dispatch.pop("definition")
    first_dispatch.pop("config")
    first_dispatch["buffers"][2] = {
        "name": "dst",
        "transient": "tmp",
        "position": 2,
        "kind": "output",
    }
    second_dispatch = json.loads(json.dumps(base_dispatch))
    second_dispatch.pop("definition")
    second_dispatch.pop("config")
    second_dispatch["buffers"][0] = {
        "name": "src0",
        "transient": "tmp",
        "position": 0,
        "kind": "input",
    }
    route["dispatches"] = [
        {
            "name": "first",
            "definition": definition,
            "config": json.loads(json.dumps(config)),
            "buffers": first_dispatch["buffers"],
            "scalars": first_dispatch["scalars"],
            "dispatch": first_dispatch["dispatch"],
        },
        {
            "name": "second",
            "definition": definition,
            "config": json.loads(json.dumps(config)),
            "buffers": second_dispatch["buffers"],
            "scalars": second_dispatch["scalars"],
            "dispatch": second_dispatch["dispatch"],
        },
    ]


def expect_multi_step_generation_valid():
    with tempfile.TemporaryDirectory(prefix="multi-step-route-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_route(source_root, make_multi_step_add_route)
        loom.validate_catalog(source_root)
        entries = catalog.build_entries(source_root, ["gfx1100"])
        entry_ids = [entry["id"] for entry in entries]
        expected_ids = ["add_f32_contiguous::first", "add_f32_contiguous::second"]
        if entry_ids != expected_ids:
            raise AssertionError(f"multi-step-route: expected entries {expected_ids}, got {entry_ids}")

        route_path, route, _, dispatch_definitions = route_impl.load_route(source_root, str(ROUTE_PATH))
        impl = route_impl.generate_op_router_impl([(route_path, route, dispatch_definitions)], "GGML_OP_ADD")
        if "plan->transients[0].offset = transient_cursor;" not in impl:
            raise AssertionError("multi-step-route: expected transient offset materialization")
        if "plan->transients[0].size = transient_0_size;" not in impl:
            raise AssertionError("multi-step-route: expected transient size materialization")
        if "GGML_BACKEND_HRX_LOOM_TRANSIENT_ALIGNMENT" not in impl:
            raise AssertionError("multi-step-route: expected fixed transient alignment")
        if "plan->transient_byte_length = transient_cursor;" not in impl:
            raise AssertionError("multi-step-route: expected total transient byte length")
        if "plan->dispatches[0].transient_binding_indices[2] = 0;" not in impl:
            raise AssertionError("multi-step-route: expected transient output binding")
        if "plan->dispatches[1].transient_binding_indices[0] = 0;" not in impl:
            raise AssertionError("multi-step-route: expected transient input binding")
        if "GGML_HRX_LOOM_ROUTE_ID_ADD_F32_CONTIGUOUS__FIRST" not in impl:
            raise AssertionError("multi-step-route: expected step catalog entry id")


def expect_undeclared_transient_invalid():
    with tempfile.TemporaryDirectory(prefix="undeclared-transient-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def mutator(route):
            make_multi_step_add_route(route)
            route["dispatches"][0]["buffers"][2]["transient"] = "missing"

        mutate_route(source_root, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if "transient buffer missing is not declared" in message:
                return
            raise AssertionError(f"undeclared-transient: unexpected error {message!r}") from err
        raise AssertionError("undeclared-transient: validator accepted invalid route")


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
        lambda route: route["dispatches"][0].update({
            "dispatch": {"workgroups": ["derived.total_size"], "workgroup_size": [256, 1, 1]}
        }),
    )
    expect_dependency_generation_valid()
    expect_index_scalar_valid()
    expect_multi_step_generation_valid()
    expect_undeclared_transient_invalid()

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
            lambda route: route["dispatches"][0]["dispatch"].update({"workgroups": ["derived.total_size"]}),
            "expects exactly one of work_items or workgroups",
        ),
        (
            "scalar-dispatch",
            lambda route: route["dispatches"][0]["dispatch"].update({"work_items": "derived.total_size"}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "too-many-dispatch-axes",
            lambda route: route["dispatches"][0]["dispatch"].update({"work_items": [1, 1, 1, 1]}),
            "expected an array with 1 to 3 integer values",
        ),
        (
            "missing-config",
            lambda route: route["dispatches"][0].pop("config"),
            "expected object field config",
        ),
        (
            "duplicate-config-names",
            lambda route: route["dispatches"][0]["config"]["bindings"][1].update({"name": "shape_pointwise_total_size"}),
            "duplicate config binding name shape_pointwise_total_size",
        ),
        (
            "unresolved-config-source",
            lambda route: route["dispatches"][0]["config"]["bindings"][0].update({"source": "derived.missing"}),
            "derived value missing is not available",
        ),
        (
            "source-value-conflict",
            lambda route: route["dispatches"][0]["config"]["bindings"][0].update({"value": 7}),
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
        "fusion-unknown-elided",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"][2].update({"elided": ["missing"]}),
        "tensor missing is not declared in tensors",
    )
    expect_invalid_full_catalog_mutation(
        "fusion-output-elided",
        FUSION_ROUTE_PATH,
        lambda route: route["dispatches"][0]["buffers"][2].update({"tensor": "rms_out"}),
        "elided tensor rms_out cannot be bound as a dispatch buffer",
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
        "empty-definition-dependencies",
        lambda source_root: set_definition_dependencies(source_root, []),
        "dependencies must not be empty",
    )
    expect_invalid_catalog_mutation(
        "scalar-definition-dependencies",
        lambda source_root: mutate_definition(source_root, lambda definition: definition.update({"dependencies": 7})),
        "expected array field dependencies",
    )
    expect_invalid_catalog_mutation(
        "non-object-definition-dependency",
        lambda source_root: set_definition_dependencies(source_root, [7]),
        "expected object",
    )
    expect_invalid_catalog_mutation(
        "unknown-definition-dependency-field",
        lambda source_root: (
            set_definition_dependencies(
                source_root,
                [{"source": add_dependency_source(source_root), "source_format": "loom-text", "extra": True}],
            )
        ),
        "unsupported field extra",
    )
    expect_invalid_catalog_mutation(
        "empty-definition-dependency-source",
        lambda source_root: set_definition_dependencies(source_root, [{"source": "", "source_format": "loom-text"}]),
        "expected non-empty string field source",
    )
    expect_invalid_catalog_mutation(
        "empty-definition-dependency-format",
        lambda source_root: set_definition_dependencies(
            source_root,
            [{"source": add_dependency_source(source_root), "source_format": ""}],
        ),
        "expected non-empty string field source_format",
    )
    expect_invalid_catalog_mutation(
        "missing-definition-dependency-source",
        lambda source_root: set_definition_dependencies(
            source_root,
            [{"source": "sources/generic/add/f32/missing.loom", "source_format": "loom-text"}],
        ),
        "missing source",
    )
    expect_invalid_catalog_mutation(
        "unsupported-definition-dependency-format",
        lambda source_root: set_definition_dependencies(
            source_root,
            [{"source": add_dependency_source(source_root), "source_format": "amdgpu-hsaco"}],
        ),
        "unsupported source_format amdgpu-hsaco; expected one of loom-text",
    )
    expect_invalid_catalog_mutation(
        "duplicate-definition-dependency-source",
        lambda source_root: (
            lambda dependency_source: set_definition_dependencies(
                source_root,
                [
                    {"source": dependency_source, "source_format": "loom-text"},
                    {"source": f"./{dependency_source}", "source_format": "loom-text"},
                ],
            )
        )(add_dependency_source(source_root)),
        "duplicate dependency source",
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
