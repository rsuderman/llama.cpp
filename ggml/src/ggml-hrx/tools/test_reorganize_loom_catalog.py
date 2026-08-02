#!/usr/bin/env python3

import json
import sys
import tempfile
from pathlib import Path

from utils import hrx_catalog_rebucket


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_catalog(root):
    write_json(root / "metadata.json", {
        "schema": "ggml-hrx-loom-catalog-v0",
        "version": 0,
        "targets": ["gfx1100"],
        "routes": ["routes/gfx1100/qwen_moe/test.json"],
    })
    write_text(
        root / "sources/gfx1100/qwen_moe/test.loom",
        "amdgpu.target<gfx11-generic> @test_wave32 {subgroup_size = 32}\n",
    )
    write_text(root / "sources/gfx1100/qwen_moe/helper.loom", "func.def inline @helper() {}\n")
    write_text(
        root / "sources/gfx1100/qwen_moe/exact.loom",
        "amdgpu.target<gfx1100> @exact_wave32 {subgroup_size = 32}\n",
    )
    write_text(root / "sources/gfx1100/swiglu/f32/qwen3_moe_common.loom", "func.def inline @qwen3_moe_common() {}\n")
    write_text(root / "sources/generic/helper.loom", "func.def @generic_helper() {}\n")
    write_json(root / "defs/generic/helper.json", {
        "schema": "ggml-hrx-loom-def-v1",
        "id": "generic_helper",
        "op": "GGML_OP_MUL",
        "source": "sources/generic/helper.loom",
        "source_format": "loom-text",
        "symbol": "generic_helper",
        "abi": {"parameter_count": 0, "constant_byte_length": 0},
        "workgroup_size": [1, 1, 1],
    })
    write_json(root / "defs/gfx1100/qwen_moe/test.json", {
        "schema": "ggml-hrx-loom-def-v1",
        "id": "test",
        "op": "GGML_OP_MUL",
        "source": "sources/gfx1100/qwen_moe/test.loom",
        "source_format": "loom-text",
        "symbol": "test",
        "abi": {"parameter_count": 0, "constant_byte_length": 0},
        "workgroup_size": [32, 1, 1],
        "dependencies": [
            {"source": "sources/gfx1100/qwen_moe/helper.loom", "source_format": "loom-text"},
            {"source": "sources/gfx1100/qwen_moe/exact.loom", "source_format": "loom-text"},
            {"source": "sources/gfx1100/swiglu/f32/qwen3_moe_common.loom", "source_format": "loom-text"},
        ],
    })
    write_json(root / "routes/gfx1100/qwen_moe/test.json", {
        "schema": "ggml-hrx-loom-route-v1",
        "id": "test_route",
        "format": "loom",
        "priority": 1,
        "match": {"op": "GGML_OP_MUL", "tensors": {}, "attributes": {}},
        "derived": {},
        "architectures": ["gfx1100"],
        "dispatches": [
            {"name": "generic_helper", "definition": "../../../defs/generic/helper.json"},
            {"name": "test", "definition": "../../../defs/gfx1100/qwen_moe/test.json"},
        ],
    })


def expect_gfx11_generic_rebucket():
    with tempfile.TemporaryDirectory(prefix="loom-rebucket-") as tmpdir:
        root = Path(tmpdir) / "loom-catalog"
        make_catalog(root)
        plan = hrx_catalog_rebucket.collect_plan(root)
        if plan.issues:
            raise AssertionError(f"rebucket: unexpected issues {plan.issues}")
        move_paths = {(move.old_path.as_posix(), move.new_path.as_posix()) for move in plan.moves}
        expected_moves = {
            ("sources/gfx1100/qwen_moe/test.loom", "sources/gfx11-generic/qwen_moe/test.loom"),
            ("sources/gfx1100/qwen_moe/helper.loom", "sources/gfx11-generic/qwen_moe/helper.loom"),
            ("sources/gfx1100/qwen_moe/exact.loom", "sources/gfx11-generic/qwen_moe/exact.loom"),
            ("sources/gfx1100/swiglu/f32/qwen3_moe_common.loom", "sources/gfx11-generic/swiglu/f32/qwen3_moe_common.loom"),
            ("defs/gfx1100/qwen_moe/test.json", "defs/gfx11-generic/qwen_moe/test.json"),
            ("routes/gfx1100/qwen_moe/test.json", "routes/gfx11-generic/qwen_moe/test.json"),
        }
        if move_paths != expected_moves:
            raise AssertionError(f"rebucket: expected moves {expected_moves}, got {move_paths}")
        hrx_catalog_rebucket.apply_plan(root, plan)
        metadata = read_json(root / "metadata.json")
        route = read_json(root / "routes/gfx11-generic/qwen_moe/test.json")
        definition = read_json(root / "defs/gfx11-generic/qwen_moe/test.json")
        if metadata["targets"] != ["gfx1100", "gfx1151"]:
            raise AssertionError("rebucket: metadata targets must stay exact and include gfx11 hardware")
        if metadata["routes"] != ["routes/gfx11-generic/qwen_moe/test.json"]:
            raise AssertionError("rebucket: metadata route did not move to gfx11-generic")
        if route["architectures"] != ["gfx1100", "gfx1151"]:
            raise AssertionError("rebucket: route architectures must list exact gfx11 hardware")
        if route["dispatches"][0]["definition"] != "../../../defs/generic/helper.json":
            raise AssertionError("rebucket: generic route definition should stay generic")
        if route["dispatches"][1]["definition"] != "../../../defs/gfx11-generic/qwen_moe/test.json":
            raise AssertionError("rebucket: route definition did not point at gfx11-generic def")
        if definition["source"] != "sources/gfx11-generic/qwen_moe/test.loom":
            raise AssertionError("rebucket: def source did not point at gfx11-generic source")
        if definition["dependencies"][0]["source"] != "sources/gfx11-generic/qwen_moe/helper.loom":
            raise AssertionError("rebucket: qwen_moe targetless helper did not move to gfx11-generic")
        if definition["dependencies"][1]["source"] != "sources/gfx11-generic/qwen_moe/exact.loom":
            raise AssertionError("rebucket: qwen_moe exact target helper did not move to gfx11-generic")
        if definition["dependencies"][2]["source"] != "sources/gfx11-generic/swiglu/f32/qwen3_moe_common.loom":
            raise AssertionError("rebucket: qwen3_moe named helper did not move to gfx11-generic")
        if "amdgpu.target<gfx11-generic>" not in (root / "sources/gfx11-generic/qwen_moe/exact.loom").read_text(encoding="utf-8"):
            raise AssertionError("rebucket: qwen_moe exact target source did not use gfx11-generic target")


def expect_mixed_targets_invalid():
    with tempfile.TemporaryDirectory(prefix="loom-rebucket-mixed-") as tmpdir:
        root = Path(tmpdir) / "loom-catalog"
        make_catalog(root)
        write_text(
            root / "sources/gfx1100/qwen_moe/test.loom",
            "amdgpu.target<gfx11-generic> @a {subgroup_size = 32}\n"
            "amdgpu.target<gfx1100> @b {subgroup_size = 32}\n",
        )
        plan = hrx_catalog_rebucket.collect_plan(root)
        if not any("mixed AMDGPU targets" in issue for issue in plan.issues):
            raise AssertionError(f"mixed-targets: expected mixed target issue, got {plan.issues}")


def expect_metadata_target_invalid():
    with tempfile.TemporaryDirectory(prefix="loom-rebucket-metadata-") as tmpdir:
        root = Path(tmpdir) / "loom-catalog"
        make_catalog(root)
        metadata = read_json(root / "metadata.json")
        metadata["targets"] = ["gfx11-generic"]
        write_json(root / "metadata.json", metadata)
        plan = hrx_catalog_rebucket.collect_plan(root)
        if not any("metadata targets must be exact hardware" in issue for issue in plan.issues):
            raise AssertionError(f"metadata-target: expected exact hardware issue, got {plan.issues}")


def main():
    expect_gfx11_generic_rebucket()
    expect_mixed_targets_invalid()
    expect_metadata_target_invalid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
