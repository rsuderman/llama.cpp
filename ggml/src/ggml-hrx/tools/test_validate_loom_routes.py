#!/usr/bin/env python3

import copy
import json
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import generate_loom_catalog as catalog
import generate_loom_route_impl as route_impl
import update_loom_route_priorities as priority_update
import validate_loom_routes as loom
from utils import hrx_route_priority


TOOLS_DIR = Path(__file__).resolve().parent
CATALOG_ROOT = TOOLS_DIR.parent / "loom-catalog"
METADATA_PATH = Path("metadata.json")
ROUTE_PATH = Path("routes/generic/add/f32/contiguous.json")
SUM_ROWS_ROUTE_PATH = Path("routes/generic/sum_rows/f32/contiguous_4d.json")
FUSION_ROUTE_PATH = Path("routes/generic/rms_norm_mul/f32/contiguous_4d.json")
RECURRENT_ROUTE_PATH = Path(
    "routes/gfx1151/gated_delta_net/f32/state_cache_decode_token1_compound.json"
)
RECURRENT_SHARED_EMPTY_INDEX_ROUTE_PATH = Path(
    "routes/gfx1151/gated_delta_net/f32/"
    "state_cache_decode_token1_compound_shared_empty_index.json"
)
RECURRENT_ZERO_SCALE_ROUTE_PATH = Path(
    "routes/gfx1151/gated_delta_net/f32/"
    "state_cache_decode_token1_compound_zero_scale.json"
)
RECURRENT_SHARED_EMPTY_INDEX_ZERO_SCALE_ROUTE_PATH = Path(
    "routes/gfx1151/gated_delta_net/f32/"
    "state_cache_decode_token1_compound_shared_empty_index_zero_scale.json"
)
CONCAT_WINDOW_TAIL_ROUTE_PATH = Path(
    "routes/gfx1151/concat/f32/window_tail_ssm_silu_pp512.json"
)
CONCAT_WINDOW_TAIL_ZERO_SCALE_ROUTE_PATH = Path(
    "routes/gfx1151/concat/f32/window_tail_ssm_silu_pp512_zero_scale.json"
)
PP_GDN_RMS_SIDE_ROUTE_PATH = Path(
    "routes/gfx1151/gated_delta_net/f32/"
    "sv128_qk_l2_full_head_rms_scale_fused.json"
)
Q5_DOWN_STORAGE_TRANSFORM_PATH = Path(
    "storage-transforms/gfx1151/q5_k_expert_down_group4.json"
)
Q5_DOWN_STORAGE_CONSUMERS = {
    Path(
        "sources/gfx1151/mul_mat_id/q5_k_f32/"
        "mul_mat_id_q5_k_f32_mmqt_down_group4.loom"
    ): ("c0_0_k", "c0_1_k", "c1_0_k", "c1_1_k"),
    Path(
        "sources/gfx1151/mul_mat_id/q5_k_f32/"
        "mul_mat_id_q5_k_f32_mmqt_down_group4_tableless_decode.loom"
    ): ("c0_0_k", "c0_1_k", "c1_0_k", "c1_1_k"),
    Path(
        "sources/gfx1151/mul_mat_id/q5_k_f32/"
        "mul_mat_id_q5_k_f32_mmqt_down_group4_terminal_qact.loom"
    ): ("rd_p0_q0_c", "rd_p0_q1_c", "rd_p1_q0_c", "rd_p1_q1_c"),
    Path(
        "sources/gfx1151/mul_mat_id/q5_k_f32/"
        "q5_down_tableless_compact_qact_decode.loom"
    ): ("rd_p0_q0_c", "rd_p0_q1_c", "rd_p1_q0_c", "rd_p1_q1_c"),
    Path("sources/gfx1151/graph/q5_down_f16_pp512.loom"): (
        "rd_p0_q0_c",
        "rd_p0_q1_c",
        "rd_p1_q0_c",
        "rd_p1_q1_c",
    ),
}
FA_VARIABLE_KV_ROUTE_PATHS = (
    Path("routes/gfx1151/flash_attn_ext/f32_f16/wmma_gate_epilogue.json"),
    Path("routes/gfx1151/flash_attn_ext/f32_f16/direct_kv64_f32acc_gate.json"),
)
FA_DIRECT_SOURCE_PATH = Path(
    "sources/gfx1151/flash_attn_ext/f32_f16/direct_kv64_f32acc_gate.loom"
)
RUNTIME_PUBLIC_HEADER = CATALOG_ROOT / "ggml-hrx-loom-catalog-runtime.h"
RUNTIME_INTERNAL_HEADER = CATALOG_ROOT / "ggml-hrx-loom-catalog-runtime-internal.h"
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


def expect_derived_math_valid():
    with tempfile.TemporaryDirectory(prefix="derived-math-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def add_derived_math(route):
            route["derived"]["plus_padding"] = {
                "type": "i64",
                "sum": ["derived.total_size", 1, 3],
            }
            route["derived"]["without_padding"] = {
                "type": "i64",
                "difference": ["derived.plus_padding", 4],
            }
            route["derived"]["max_extent"] = {
                "type": "i64",
                "maximum": ["derived.total_size", 32],
            }

        mutate_route(source_root, add_derived_math)
        loom.validate_catalog(source_root)

        route, definition = read_route_and_definition(source_root, ROUTE_PATH)
        impl = route_impl.generate_route_impl(str(ROUTE_PATH), route, definition)
        if (
            "const int64_t derived_plus_padding = static_cast<int64_t>(derived_total_size) "
            "+ static_cast<int64_t>(1) + static_cast<int64_t>(3);"
        ) not in impl:
            raise AssertionError("derived-math: expected sum expression")
        if (
            "const int64_t derived_without_padding = static_cast<int64_t>(derived_plus_padding) "
            "- static_cast<int64_t>(4);"
        ) not in impl:
            raise AssertionError("derived-math: expected difference expression")
        if (
            "const int64_t derived_max_extent = "
            "std::max(static_cast<int64_t>(derived_total_size), static_cast<int64_t>(32));"
        ) not in impl:
            raise AssertionError("derived-math: expected maximum expression")


def route_for_priority(schema, route_id, ops=None):
    route = {
        "schema": schema,
        "id": route_id,
        "format": "loom",
        "priority": 1,
        "derived": {},
        "architectures": ["gfx1100"],
        "dispatches": [{"name": "dispatch"}],
    }
    if ops is None:
        route["match"] = {"op": "GGML_OP_MUL", "tensors": {}, "attributes": {}}
    else:
        route["match"] = {
            "ops": {f"op{i}": {"op": op, "tensors": {}, "attributes": {}} for i, op in enumerate(ops)},
            "anchors": ["op0"],
            "predicates": [{"field": "derived.n", "equals": 1}],
        }
    return route


def expect_route_priority_order():
    model_fusion = hrx_route_priority.route_priority(
        "routes/gfx1100/test_family/test_model/big.json",
        route_for_priority(
            "ggml-hrx-loom-fusion-route-v2",
            "test_model_big",
            ["GGML_OP_MUL"] * 10,
        ),
    )
    gfx_specific = hrx_route_priority.route_priority(
        "routes/gfx1100/mul_mat/f32_f32/tiled_gfx1100.json",
        route_for_priority("ggml-hrx-loom-route-v1", "mul_mat_tiled_gfx1100"),
    )
    gfx_generic = hrx_route_priority.route_priority(
        "routes/gfx1100/flash_attn_ext/f32_f16/tiled_decode.json",
        route_for_priority("ggml-hrx-loom-route-v1", "flash_attn_ext_tiled_decode"),
    )
    generic_fusion = hrx_route_priority.route_priority(
        "routes/generic/mul/f32/mul_add.json",
        route_for_priority(
            "ggml-hrx-loom-fusion-route-v2",
            "mul_add",
            ["GGML_OP_MUL", "GGML_OP_ADD"],
        ),
    )

    if not model_fusion.priority > gfx_specific.priority > gfx_generic.priority > generic_fusion.priority:
        raise AssertionError("route-priority: expected model fusion > gfx specific > gfx generic > generic")


def expect_terminal_output_transients_raise_priority():
    base = route_for_priority(
        "ggml-hrx-loom-fusion-route-v2",
        "test_route_a",
        ["GGML_OP_MUL", "GGML_OP_ADD", "GGML_OP_MUL"],
    )
    terminal = route_for_priority(
        "ggml-hrx-loom-fusion-route-v2",
        "test_route_b",
        ["GGML_OP_MUL", "GGML_OP_ADD", "GGML_OP_MUL"],
    )
    terminal["dispatches"] = [
        {
            "name": "dispatch",
            "buffers": [
                {"name": "internal", "kind": "output", "transient": "tmp0"},
            ],
        },
        {
            "name": "dispatch",
            "buffers": [
                {"name": "internal", "kind": "input", "transient": "tmp0"},
                {"name": "published", "kind": "output", "transient": "tmp1"},
            ],
        },
    ]

    base_priority = hrx_route_priority.route_priority("routes/gfx1100/test_family/test_model/a.json", base)
    terminal_priority = hrx_route_priority.route_priority("routes/gfx1100/test_family/test_model/b.json", terminal)

    if terminal_priority.priority <= base_priority.priority:
        raise AssertionError("route-priority: expected terminal output transient to raise priority")
    if "terminal output transients: 1" not in terminal_priority.reasons:
        raise AssertionError("route-priority: expected terminal output transient reason")


def expect_priority_replacement_preserves_route_text():
    with tempfile.TemporaryDirectory(prefix="priority-replace-") as tmpdir:
        route_path = Path(tmpdir) / "route.json"
        route_path.write_text('{\n  "id": "test",\n  "priority": 7,\n  "other": 1\n}\n', encoding="utf-8")
        if not priority_update.replace_priority_text(route_path, 42):
            raise AssertionError("priority-replace: expected priority update")
        text = route_path.read_text(encoding="utf-8")
        if text != '{\n  "id": "test",\n  "priority": 42,\n  "other": 1\n}\n':
            raise AssertionError("priority-replace: unexpected text rewrite")


def expect_view_field_predicates_valid():
    with tempfile.TemporaryDirectory(prefix="view-field-predicates-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def add_view_predicates(route):
            route["match"]["predicates"].extend([
                {"field": "tensor.dst.view_source", "equals": "tensor.src0"},
                {"field": "tensor.dst.view_offset_bytes", "equals": 0},
            ])

        mutate_route(source_root, add_view_predicates)
        loom.validate_catalog(source_root)

        route, definition = read_route_and_definition(source_root, ROUTE_PATH)
        impl = route_impl.generate_route_impl(str(ROUTE_PATH), route, definition)
        if "(dst->view_src ? dst->view_src : nullptr) == src0" not in impl:
            raise AssertionError("view-field-predicates: expected view source comparison")
        if "static_cast<int64_t>(dst->view_offs) == 0" not in impl:
            raise AssertionError("view-field-predicates: expected view offset comparison")


def expect_fusion_reshape_valid():
    with tempfile.TemporaryDirectory(prefix="fusion-reshape-") as tmpdir:
        source_root = copy_catalog(tmpdir)
        mutate_metadata(source_root, lambda metadata: metadata.update({"routes": [str(FUSION_ROUTE_PATH)]}))

        def add_reshape(route):
            route["tensors"]["rms_view"] = json.loads(json.dumps(route["tensors"]["rms_out"]))
            route["match"]["ops"] = {
                "rms": route["match"]["ops"]["rms"],
                "reshape": {
                    "op": "GGML_OP_RESHAPE",
                    "tensors": {"src0": "rms_out", "dst": "rms_view"},
                    "attributes": {},
                },
                "mul": route["match"]["ops"]["mul"],
            }
            route["match"]["ops"]["mul"]["tensors"]["src0"] = "rms_view"
            route["match"]["predicates"][2] = {"elided": ["rms_out", "rms_view"]}

        mutate_route_at(source_root, FUSION_ROUTE_PATH, add_reshape)
        loom.validate_catalog(source_root)

        route, definition = read_route_and_definition(source_root, FUSION_ROUTE_PATH)
        impl = route_impl.generate_route_impl(str(FUSION_ROUTE_PATH), route, definition)
        if "reshape_op->op != GGML_OP_RESHAPE" not in impl:
            raise AssertionError("fusion-reshape: expected reshape op check")


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


def expect_graph_transient_generation_valid():
    with tempfile.TemporaryDirectory(prefix="graph-transient-route-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def mutator(route):
            make_multi_step_add_route(route)
            route.pop("transient_buffers")
            route["dispatches"][0]["buffers"][2] = {
                "name": "dst",
                "tensor": "dst",
                "storage": "graph_transient",
                "position": 2,
                "kind": "output",
            }
            route["dispatches"][1]["buffers"][0] = {
                "name": "src0",
                "tensor": "dst",
                "storage": "graph_transient",
                "position": 0,
                "kind": "input",
            }

        mutate_route(source_root, mutator)
        loom.validate_catalog(source_root)

        route_path, route, _, dispatch_definitions = route_impl.load_route(source_root, str(ROUTE_PATH))
        impl = route_impl.generate_op_router_impl([(route_path, route, dispatch_definitions)], "GGML_OP_ADD")
        if "plan->transients[0].graph_tensor = dst;" not in impl:
            raise AssertionError("graph-transient-route: expected graph tensor materialization")
        if "plan->dispatches[0].transient_binding_accesses[2] = GGML_BACKEND_HRX_LOOM_BUFFER_ACCESS_WRITE;" not in impl:
            raise AssertionError("graph-transient-route: expected graph transient write binding")
        if "plan->dispatches[1].transient_binding_accesses[0] = GGML_BACKEND_HRX_LOOM_BUFFER_ACCESS_READ;" not in impl:
            raise AssertionError("graph-transient-route: expected graph transient read binding")


def expect_transient_storage_invalid():
    with tempfile.TemporaryDirectory(prefix="graph-transient-invalid-") as tmpdir:
        source_root = copy_catalog(tmpdir)

        def mutator(route):
            make_multi_step_add_route(route)
            route["dispatches"][0]["buffers"][2]["storage"] = "graph_transient"

        mutate_route(source_root, mutator)
        try:
            loom.validate_catalog(source_root)
        except ValueError as err:
            message = str(err)
            if "storage is only supported for tensor buffers" in message:
                return
            raise AssertionError(f"graph-transient-invalid: unexpected error {message!r}") from err
        raise AssertionError("graph-transient-invalid: validator accepted invalid route")


def fusion_route_with_consumed_count(route, count):
    result = copy.deepcopy(route)
    for i in range(len(result["match"]["ops"]), count):
        result["match"]["ops"][f"capacity_view_{i}"] = {
            "op": "GGML_OP_VIEW",
            "tensors": {"src0": "y", "dst": "y"},
            "attributes": {},
        }
    return result


def fusion_route_with_binding_count(route, definition, count):
    result_route = copy.deepcopy(route)
    result_definition = copy.deepcopy(definition)
    dispatch = result_route["dispatches"][0]
    for i in range(len(dispatch["buffers"]), count):
        name = f"capacity_input_{i}"
        dispatch["buffers"].append({
            "name": name,
            "tensor": "x",
            "position": i,
            "kind": "input",
        })
        result_definition["bindings"].append({"name": name, "access": "read"})
    result_definition["abi"]["binding_count"] = count
    result_definition["abi"]["parameter_count"] = (
        count + len(result_definition["parameters"])
    )
    return result_route, result_definition


def expect_native_fusion_capacity_boundaries():
    if route_impl.MAX_CONSUMED_NODES != 40:
        raise AssertionError("route generator consumed-node capacity must be exactly 40")
    if route_impl.MAX_BINDINGS != 16:
        raise AssertionError("route generator binding capacity must be exactly 16")

    public_header = RUNTIME_PUBLIC_HEADER.read_text(encoding="utf-8")
    if "GGML_BACKEND_HRX_LOOM_MAX_CONSUMED_NODES = 40;" not in public_header:
        raise AssertionError("public runtime consumed-node capacity must be exactly 40")
    internal_header = RUNTIME_INTERNAL_HEADER.read_text(encoding="utf-8")
    if "GGML_BACKEND_HRX_LOOM_MAX_BINDINGS        = 16;" not in internal_header:
        raise AssertionError("internal runtime binding capacity must be exactly 16")

    route_path = CATALOG_ROOT / FUSION_ROUTE_PATH
    base_route, base_definition = read_route_and_definition(
        CATALOG_ROOT, FUSION_ROUTE_PATH
    )
    accepted_consumed = fusion_route_with_consumed_count(base_route, 40)
    accepted_impl = route_impl.generate_route_impl(
        route_path, accepted_consumed, base_definition
    )
    if "static constexpr int matched_node_count = 40;" not in accepted_impl:
        raise AssertionError("generator did not accept exactly 40 matched nodes")
    if "plan->consumed_node_count = 40;" not in accepted_impl:
        raise AssertionError("generator did not materialize exactly 40 consumed nodes")

    rejected_consumed = fusion_route_with_consumed_count(base_route, 41)
    try:
        route_impl.generate_route_impl(route_path, rejected_consumed, base_definition)
    except ValueError as err:
        if "route consumes too many graph nodes: 41" not in str(err):
            raise AssertionError(f"unexpected 41-node rejection: {err}") from err
    else:
        raise AssertionError("generator accepted 41 consumed nodes")

    definition_path = (
        route_path.parent / base_route["dispatches"][0]["definition"]
    ).resolve()
    accepted_bindings, accepted_definition = fusion_route_with_binding_count(
        base_route, base_definition, 16
    )
    loom.validate_definition(accepted_definition, definition_path, CATALOG_ROOT)
    accepted_binding_impl = route_impl.generate_route_impl(
        route_path, accepted_bindings, accepted_definition
    )
    if "plan->dispatches[0].binding_count = 16;" not in accepted_binding_impl:
        raise AssertionError("generator did not accept exactly 16 bindings")

    rejected_bindings, rejected_definition = fusion_route_with_binding_count(
        base_route, base_definition, 17
    )
    loom.validate_definition(rejected_definition, definition_path, CATALOG_ROOT)
    try:
        route_impl.generate_route_impl(
            route_path, rejected_bindings, rejected_definition
        )
    except ValueError as err:
        if "route has too many bindings: 17" not in str(err):
            raise AssertionError(f"unexpected 17-binding rejection: {err}") from err
    else:
        raise AssertionError("generator accepted 17 bindings")


def expect_q5_down_storage_consumer_offsets():
    transform = read_json(CATALOG_ROOT / Q5_DOWN_STORAGE_TRANSFORM_PATH)
    layout = transform["transform"]
    component_bytes = layout["row_group"] * layout["unit_bytes"]
    expected_offsets = tuple(component_bytes * i for i in range(1, 5))
    for source_path, names in Q5_DOWN_STORAGE_CONSUMERS.items():
        source = (CATALOG_ROOT / source_path).read_text(encoding="utf-8")
        for name, offset in zip(names, expected_offsets):
            marker = f"%{name} = index.constant {offset} : index"
            if marker not in source:
                raise AssertionError(
                    f"{source_path}: missing row-group component offset {marker}"
                )


def expect_variable_kv_fa_mask_extent_guards():
    expected_extent = {
        "type": "i64",
        "product": [
            "shape.q.ntokens",
            "tensor.mask.element_strides.1",
        ],
    }
    expected_guard = {
        "field": "derived.mask_token_extent",
        "max": 268435456,
    }
    expected_row_guard = {
        "field": "shape.mask.ntokens",
        "min": "shape.q.ntokens",
    }
    for route_path in FA_VARIABLE_KV_ROUTE_PATHS:
        route = read_json(CATALOG_ROOT / route_path)
        if route["derived"].get("mask_token_extent") != expected_extent:
            raise AssertionError(f"{route_path}: missing mask token extent")
        if expected_guard not in route["match"]["predicates"]:
            raise AssertionError(f"{route_path}: missing mask token extent guard")
        if expected_row_guard not in route["match"]["predicates"]:
            raise AssertionError(f"{route_path}: missing mask row count guard")

    direct_route = read_json(CATALOG_ROOT / FA_VARIABLE_KV_ROUTE_PATHS[1])
    direct_dispatch = direct_route["dispatches"][0]
    expected_runtime_scalars = {
        "key_value_token_count": ("index", 0, "shape.k.nkv"),
        "mask_stride_token": ("index", 1, "tensor.mask.element_strides.1"),
        "scale": ("f32", 2, "attribute.fa.scale"),
    }
    runtime_scalars = {
        scalar["name"]: (scalar["type"], scalar["position"], scalar["source"])
        for scalar in direct_dispatch["scalars"]
    }
    for name, expected in expected_runtime_scalars.items():
        if runtime_scalars.get(name) != expected:
            raise AssertionError(f"direct FA route has invalid runtime scalar {name}")
    compile_bindings = {
        binding["name"] for binding in direct_dispatch["config"]["bindings"]
    }
    if {
        "hrx2_shape_fa_nkv",
        "hrx2_shape_fa_mask_stride_token",
    } & compile_bindings:
        raise AssertionError("direct FA route specializes variable KV extents")
    expected_predicates = (
        {"field": "shape.k.nkv", "min": 512},
        {"field": "shape.k.nkv", "max": 32768},
        {"field": "shape.k.nkv", "multiple_of": 64},
        {"field": "shape.mask.nkv", "min": "shape.k.nkv"},
        {
            "field": "tensor.mask.element_strides.1",
            "equals": "shape.mask.nkv",
        },
    )
    for predicate in expected_predicates:
        if predicate not in direct_route["match"]["predicates"]:
            raise AssertionError(f"direct FA route is missing predicate {predicate}")

    definition_path = (
        CATALOG_ROOT
        / FA_VARIABLE_KV_ROUTE_PATHS[1].parent
        / direct_dispatch["definition"]
    ).resolve()
    definition = read_json(definition_path)
    if definition["abi"] != {
        "binding_count": 6,
        "parameter_count": 9,
        "constant_byte_length": 12,
    }:
        raise AssertionError("direct FA runtime ABI counts are invalid")
    if definition["parameters"] != [
        {"name": "key_value_token_count", "type": "index"},
        {"name": "mask_stride_token", "type": "index"},
        {"name": "scale", "type": "f32"},
    ]:
        raise AssertionError("direct FA runtime ABI parameter order is invalid")

    source = (CATALOG_ROOT / FA_DIRECT_SOURCE_PATH).read_text(encoding="utf-8")
    expected_view = (
        "view<[%bounded_query_token_count]x[%bounded_mask_stride_token]xf16, #dense>"
    )
    expected_signature = (
        "(%key_value_token_count: index, %mask_stride_token: index)"
    )
    if expected_signature not in source:
        raise AssertionError("direct FA source does not accept runtime KV extents")
    expected_bounds = (
        "%bounded_key_value_token_count, %bounded_mask_stride_token = "
        "index.assume %key_value_token_count, %mask_stride_token"
    )
    if expected_bounds not in source or "le(%key_value_token_count, %mask_stride_token)" not in source:
        raise AssertionError("direct FA source does not relate the runtime KV extents")
    if f"%mask_view = buffer.view %mask_aligned[%zero_offset] : buffer -> {expected_view}" not in source:
        raise AssertionError("direct FA source does not use the runtime mask stride")
    if source.count(f"view.load %mask_view[") != 5:
        raise AssertionError("unexpected direct FA mask load count")
    if source.count(f": {expected_view} -> f16") != 5:
        raise AssertionError("direct FA mask loads do not use the runtime mask stride")


def expect_native_recurrent_op_schema_and_generation():
    expected_rules = {
        "GGML_OP_RESHAPE": ({"src0", "dst"}, set()),
        "GGML_OP_CONCAT": ({"src0", "src1", "dst"}, set()),
        "GGML_OP_CONT": ({"src0", "dst"}, set()),
        "GGML_OP_SSM_CONV": ({"src0", "src1", "dst"}, set()),
        "GGML_OP_UNARY": ({"src0", "dst"}, set()),
        "GGML_OP_L2_NORM": ({"src0", "dst"}, set()),
        "GGML_OP_GATED_DELTA_NET": (
            {"src0", "src1", "src2", "src3", "src4", "src5", "dst"},
            set(),
        ),
        "GGML_OP_CPY": ({"src0", "dst"}, {"src1"}),
    }
    for op, (required, optional) in expected_rules.items():
        rule = loom.route_schema.OP_RULES.get(op)
        if rule is None:
            raise AssertionError(f"missing native fusion op schema for {op}")
        if rule["required_tensors"] != required:
            raise AssertionError(f"wrong required tensor schema for {op}")
        if rule["optional_tensors"] != optional:
            raise AssertionError(f"wrong optional tensor schema for {op}")

    expected_attribute_indices = {
        "GGML_OP_CONCAT": {"dim": 0},
        "GGML_OP_UNARY": {"unary_op": 0},
        "GGML_OP_L2_NORM": {"eps": 0},
        "GGML_OP_GATED_DELTA_NET": {"K": 0},
    }
    for op, indices in expected_attribute_indices.items():
        if route_impl.ATTRIBUTE_INDICES.get(op) != indices:
            raise AssertionError(f"wrong native fusion attribute indices for {op}")

    zero_extra_ops = {
        "empty_index_view",
        "conv_empty_get",
        "conv_empty_target",
        "conv_empty_write",
        "gdn_empty_get",
        "gdn_empty_target",
        "gdn_empty_write",
    }
    for recurrent_path in (
        RECURRENT_ROUTE_PATH,
        RECURRENT_SHARED_EMPTY_INDEX_ROUTE_PATH,
        RECURRENT_ZERO_SCALE_ROUTE_PATH,
        RECURRENT_SHARED_EMPTY_INDEX_ZERO_SCALE_ROUTE_PATH,
    ):
        recurrent_route = read_json(CATALOG_ROOT / recurrent_path)
        retained_zero_extra_ops = zero_extra_ops.intersection(
            recurrent_route["match"]["ops"]
        )
        if retained_zero_extra_ops:
            raise AssertionError(
                f"native recurrent route retains zero-row extra-state ops: "
                f"{sorted(retained_zero_extra_ops)}"
            )

    removed_concat_ops = {"extra_get", "extra_dst_view", "extra_copy"}
    removed_concat_tensors = {
        "zero_indices",
        "zero_get_dst",
        "zero_copy_base",
        "zero_copy_target",
        "zero_copy_dst",
    }
    for concat_path in (
        CONCAT_WINDOW_TAIL_ROUTE_PATH,
        CONCAT_WINDOW_TAIL_ZERO_SCALE_ROUTE_PATH,
    ):
        concat_route = read_json(CATALOG_ROOT / concat_path)
        retained_concat_ops = removed_concat_ops.intersection(
            concat_route["match"]["ops"]
        )
        if retained_concat_ops:
            raise AssertionError(
                f"native concat route retains zero-row extra-state ops: "
                f"{sorted(retained_concat_ops)}"
            )
        retained_concat_tensors = removed_concat_tensors.intersection(
            concat_route["tensors"]
        )
        if retained_concat_tensors:
            raise AssertionError(
                f"native concat route retains zero-row extra-state tensors: "
                f"{sorted(retained_concat_tensors)}"
            )
        if len(concat_route["match"]["ops"]) != 10:
            raise AssertionError("native concat route must match exactly 10 graph ops")

    concat_zero_scale_route = read_json(
        CATALOG_ROOT / CONCAT_WINDOW_TAIL_ZERO_SCALE_ROUTE_PATH
    )
    if [dispatch["name"] for dispatch in concat_zero_scale_route["dispatches"]] != [
        "concat_window_tail_ssm_prepass",
        "gdn_state_get",
        "main",
    ]:
        raise AssertionError("zero-scale concat route must elide only the empty scale dispatch")

    route_path = CATALOG_ROOT / RECURRENT_ROUTE_PATH
    definitions = loom.load_definitions(CATALOG_ROOT)
    loom.validate_route(route_path, definitions, {"gfx1151"})
    route = read_json(route_path)
    dispatch_definitions = loom.resolve_dispatch_definitions(
        route, route_path, definitions
    )
    impl = route_impl.generate_route_impl(
        route_path, route, dispatch_definitions
    )
    for fragment in (
        "static constexpr int matched_node_count = 40;",
        "plan->consumed_node_count = 40;",
        "plan->dispatch_count = 5;",
        "plan->dispatches[0].binding_count = 6;",
        "plan->dispatches[2].binding_count = 12;",
        "plan->dispatches[3].binding_count = 8;",
        "plan->dispatches[4].binding_count = 4;",
        "plan->transient_count = 1;",
    ):
        if fragment not in impl:
            raise AssertionError(
                f"native recurrent generator missing {fragment!r}"
            )

    shared_route_path = CATALOG_ROOT / RECURRENT_SHARED_EMPTY_INDEX_ROUTE_PATH
    loom.validate_route(shared_route_path, definitions, {"gfx1151"})
    shared_route = read_json(shared_route_path)
    shared_dispatch_definitions = loom.resolve_dispatch_definitions(
        shared_route, shared_route_path, definitions
    )
    shared_impl = route_impl.generate_route_impl(
        shared_route_path, shared_route, shared_dispatch_definitions
    )
    for fragment in (
        "static constexpr int matched_node_count = 40;",
        "plan->consumed_node_count = 40;",
        "plan->dispatch_count = 5;",
        "plan->dispatches[0].binding_count = 6;",
        "plan->dispatches[2].binding_count = 12;",
        "plan->dispatches[3].binding_count = 8;",
        "plan->dispatches[4].binding_count = 4;",
        "plan->transient_count = 1;",
    ):
        if fragment not in shared_impl:
            raise AssertionError(
                f"native shared-empty-index recurrent generator missing {fragment!r}"
            )

    zero_scale_route_path = CATALOG_ROOT / RECURRENT_ZERO_SCALE_ROUTE_PATH
    loom.validate_route(zero_scale_route_path, definitions, {"gfx1151"})
    zero_scale_route = read_json(zero_scale_route_path)
    zero_scale_dispatch_definitions = loom.resolve_dispatch_definitions(
        zero_scale_route, zero_scale_route_path, definitions
    )
    zero_scale_impl = route_impl.generate_route_impl(
        zero_scale_route_path, zero_scale_route, zero_scale_dispatch_definitions
    )
    for fragment in (
        "static constexpr int matched_node_count = 40;",
        "plan->consumed_node_count = 40;",
        "plan->dispatch_count = 4;",
        "plan->dispatches[0].binding_count = 6;",
        "plan->dispatches[1].binding_count = 12;",
        "plan->dispatches[2].binding_count = 8;",
        "plan->dispatches[3].binding_count = 4;",
        "ggml_backend_hrx_loom_bind_tensor(request, gdn_cache_read, &plan->dispatches[2].bindings[5])",
        "plan->transient_count = 1;",
    ):
        if fragment not in zero_scale_impl:
            raise AssertionError(
                f"native zero-scale recurrent generator missing {fragment!r}"
            )

    shared_zero_scale_route_path = (
        CATALOG_ROOT / RECURRENT_SHARED_EMPTY_INDEX_ZERO_SCALE_ROUTE_PATH
    )
    loom.validate_route(shared_zero_scale_route_path, definitions, {"gfx1151"})
    shared_zero_scale_route = read_json(shared_zero_scale_route_path)
    shared_zero_scale_dispatch_definitions = loom.resolve_dispatch_definitions(
        shared_zero_scale_route, shared_zero_scale_route_path, definitions
    )
    shared_zero_scale_impl = route_impl.generate_route_impl(
        shared_zero_scale_route_path,
        shared_zero_scale_route,
        shared_zero_scale_dispatch_definitions,
    )
    for fragment in (
        "static constexpr int matched_node_count = 40;",
        "plan->consumed_node_count = 40;",
        "plan->dispatch_count = 4;",
        "plan->dispatches[0].binding_count = 6;",
        "plan->dispatches[1].binding_count = 12;",
        "plan->dispatches[2].binding_count = 8;",
        "plan->dispatches[3].binding_count = 4;",
        "ggml_backend_hrx_loom_bind_tensor(request, gdn_cache_read, &plan->dispatches[2].bindings[5])",
        "plan->transient_count = 1;",
    ):
        if fragment not in shared_zero_scale_impl:
            raise AssertionError(
                f"native shared-empty-index zero-scale recurrent generator missing {fragment!r}"
            )


def expect_native_pp_gdn_rms_side_generation():
    route_path = CATALOG_ROOT / PP_GDN_RMS_SIDE_ROUTE_PATH
    definitions = loom.load_definitions(CATALOG_ROOT)
    loom.validate_route(route_path, definitions, {"gfx1151"})
    route = read_json(route_path)
    dispatch_definitions = loom.resolve_dispatch_definitions(
        route, route_path, definitions
    )
    impl = route_impl.generate_route_impl(
        route_path, route, dispatch_definitions
    )
    for fragment in (
        "static constexpr int matched_node_count = 25;",
        "plan->consumed_node_count = 25;",
        "plan->dispatch_count = 5;",
        "plan->dispatches[0].binding_count = 7;",
        "plan->dispatches[1].binding_count = 8;",
        "plan->dispatches[2].binding_count = 2;",
        "plan->dispatches[3].binding_count = 4;",
        "plan->dispatches[4].binding_count = 9;",
        "plan->transient_count = 4;",
    ):
        if fragment not in impl:
            raise AssertionError(
                f"native PP GDN/RMS-side generator missing {fragment!r}"
            )


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
    expect_derived_math_valid()
    expect_route_priority_order()
    expect_terminal_output_transients_raise_priority()
    expect_priority_replacement_preserves_route_text()
    expect_view_field_predicates_valid()
    expect_fusion_reshape_valid()
    expect_multi_step_generation_valid()
    expect_undeclared_transient_invalid()
    expect_graph_transient_generation_valid()
    expect_transient_storage_invalid()
    expect_native_fusion_capacity_boundaries()
    expect_q5_down_storage_consumer_offsets()
    expect_variable_kv_fa_mask_extent_guards()
    expect_native_recurrent_op_schema_and_generation()
    expect_native_pp_gdn_rms_side_generation()

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
            "float-derived-sum",
            lambda route: route["derived"].update({"bad_sum": {"type": "f32", "sum": ["derived.total_size", 1]}}),
            "result must be an integer type",
        ),
        (
            "short-derived-difference",
            lambda route: route["derived"].update({"bad_difference": {"type": "i64", "difference": ["derived.total_size"]}}),
            "expected at least two operands",
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
    expect_invalid_full_catalog_mutation(
        "fusion-same-or-disjoint-storage-arity",
        FUSION_ROUTE_PATH,
        lambda route: route["match"]["predicates"].append(
            {"same_or_disjoint_storage": ["x"]}
        ),
        "expected at least two tensors",
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
