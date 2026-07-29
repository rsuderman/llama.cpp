#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import hrx_catalog_emit as catalog_emit


CATALOG_SCHEMA_V0 = "ggml-hrx-hsaco-catalog-v0"


require_string = catalog_emit.require_string
require_list = catalog_emit.require_list
require_int = catalog_emit.require_int
read_json = catalog_emit.read_json
c_array = catalog_emit.c_array
parse_targets = catalog_emit.parse_targets
c_identifier = catalog_emit.c_identifier


def run_command(args):
    subprocess.run(args, check=True)


def compile_hsaco(hipcc, bundler, source, target, out_dir):
    bundle = out_dir / f"{source.stem}.{target}.bundle.hsaco"
    raw = out_dir / f"{source.stem}.{target}.hsaco"
    host = out_dir / f"{source.stem}.{target}.host.o"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        hipcc,
        "--genco",
        f"--offload-arch={target}",
        "-O2",
        "-c",
        str(source),
        "-o",
        str(bundle),
    ])
    run_command([
        bundler,
        "--unbundle",
        "--type=o",
        f"--targets=hipv4-amdgcn-amd-amdhsa--{target},host-x86_64-unknown-linux-gnu-",
        f"--input={bundle}",
        f"--output={raw}",
        f"--output={host}",
    ])
    return raw


def write_generated_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_backend_hrx_hsaco_catalog_entry {
    const char * id;
    const char * op;
    const char * target;
    const char * symbol;
    const unsigned char * data;
    size_t data_size;
    uint32_t workgroup_size[3];
    uint32_t threads_per_block;
    uint32_t binding_count;
    uint32_t parameter_count;
    uint32_t constant_byte_length;
};

const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_catalog_entries(size_t * count);
""", encoding="utf-8")


def write_generated_source(path, header_path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_name = header_path.name
    chunks = [f"#include \"{header_name}\"\n\n"]
    for entry in entries:
        chunks.append(f"static const unsigned char {entry['array_name']}[] = {{\n")
        chunks.append(c_array(entry["data"]))
        chunks.append("\n};\n\n")

    chunks.append("static const ggml_backend_hrx_hsaco_catalog_entry GGML_HRX_HSACO_CATALOG[] = {\n")
    for entry in entries:
        workgroup_size = entry["workgroup_size"]
        chunks.append(
            "    {\n"
            f"        /* .id = */ \"{entry['id']}\",\n"
            f"        /* .op = */ \"{entry['op']}\",\n"
            f"        /* .target = */ \"{entry['target']}\",\n"
            f"        /* .symbol = */ \"{entry['symbol']}\",\n"
            f"        /* .data = */ {entry['array_name']},\n"
            f"        /* .data_size = */ sizeof({entry['array_name']}),\n"
            f"        /* .workgroup_size = */ {{{workgroup_size[0]}, {workgroup_size[1]}, {workgroup_size[2]}}},\n"
            f"        /* .threads_per_block = */ {entry['threads_per_block']},\n"
            f"        /* .binding_count = */ {entry['binding_count']},\n"
            f"        /* .parameter_count = */ {entry['parameter_count']},\n"
            f"        /* .constant_byte_length = */ {entry['constant_byte_length']},\n"
            "    },\n"
        )
    chunks.append("};\n\n")
    chunks.append("""const ggml_backend_hrx_hsaco_catalog_entry * ggml_backend_hrx_hsaco_catalog_entries(size_t * count) {
    if (count) {
        *count = sizeof(GGML_HRX_HSACO_CATALOG) / sizeof(GGML_HRX_HSACO_CATALOG[0]);
    }
    return GGML_HRX_HSACO_CATALOG;
}
""")
    path.write_text("".join(chunks), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--build-root", required=True)
    parser.add_argument("--out-cpp", required=True)
    parser.add_argument("--out-h", required=True)
    parser.add_argument("--hipcc", required=True)
    parser.add_argument("--bundler", required=True)
    parser.add_argument("--targets")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    build_root = Path(args.build_root)
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    if require_string(metadata, "schema", metadata_path) != CATALOG_SCHEMA_V0:
        raise ValueError(f"{metadata_path}: unsupported schema")
    targets = parse_targets(args.targets)
    if not targets:
        targets = require_list(metadata, "targets", metadata_path)
    for target in targets:
        if not isinstance(target, str) or not target:
            raise ValueError(f"{metadata_path}: targets must contain non-empty strings")

    entries = []
    for route_name in require_list(metadata, "routes", metadata_path):
        route_path = source_root / route_name
        route = read_json(route_path)
        definition_path = (route_path.parent / require_string(route, "definition", route_path)).resolve()
        definition = read_json(definition_path)
        source_path = (source_root / require_string(definition, "source", definition_path)).resolve()
        if not source_path.is_file():
            raise ValueError(f"{definition_path}: missing source {source_path}")
        launch = route.get("launch", {})
        workgroup_size = require_list(definition, "workgroup_size", definition_path)
        if len(workgroup_size) != 3:
            raise ValueError(f"{definition_path}: workgroup_size must have 3 values")
        abi = definition.get("abi")
        if not isinstance(abi, dict):
            raise ValueError(f"{definition_path}: expected object field abi")
        route_id = require_string(route, "id", route_path)
        definition_id = require_string(definition, "id", definition_path)
        for target in targets:
            hsaco_path = compile_hsaco(args.hipcc, args.bundler, source_path, target, build_root / "artifacts" / target)
            entries.append({
                "id": route_id,
                "op": require_string(definition, "op", definition_path),
                "target": target,
                "symbol": require_string(definition, "symbol", definition_path),
                "array_name": f"ggml_hrx_hsaco_{c_identifier(definition_id)}_{c_identifier(route_id)}_{c_identifier(target)}",
                "data": hsaco_path.read_bytes(),
                "workgroup_size": [int(value) for value in workgroup_size],
                "threads_per_block": int(launch.get("threads_per_block", workgroup_size[0])),
                "binding_count": require_int(abi, "binding_count", definition_path),
                "parameter_count": require_int(abi, "parameter_count", definition_path),
                "constant_byte_length": require_int(abi, "constant_byte_length", definition_path),
            })

    if not entries:
        raise ValueError(f"{metadata_path}: no routes selected for targets {targets}")

    write_generated_header(Path(args.out_h))
    write_generated_source(Path(args.out_cpp), Path(args.out_h), entries)


if __name__ == "__main__":
    main()
