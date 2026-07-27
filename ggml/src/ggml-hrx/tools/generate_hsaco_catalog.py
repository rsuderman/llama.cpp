#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


def require_string(data, key, source):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: expected non-empty string field {key}")
    return value


def require_list(data, key, source):
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: expected non-empty list field {key}")
    return value


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def c_array(data):
    rows = []
    for i in range(0, len(data), 12):
        rows.append("    " + ", ".join(f"0x{byte:02x}" for byte in data[i:i + 12]))
    return ",\n".join(rows)


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
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    build_root = Path(args.build_root)
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    if require_string(metadata, "schema", metadata_path) != "ggml-hrx-hsaco-catalog-v0":
        raise ValueError(f"{metadata_path}: unsupported schema")

    entries = []
    for route_name in require_list(metadata, "routes", metadata_path):
        route_path = source_root / route_name
        route = read_json(route_path)
        if require_string(route, "target", route_path) != args.target:
            continue
        definition_path = (route_path.parent / require_string(route, "definition", route_path)).resolve()
        definition = read_json(definition_path)
        source_path = (source_root / require_string(definition, "source", definition_path)).resolve()
        if not source_path.is_file():
            raise ValueError(f"{definition_path}: missing source {source_path}")
        hsaco_path = compile_hsaco(args.hipcc, args.bundler, source_path, args.target, build_root / "artifacts" / args.target)
        launch = route.get("launch", {})
        workgroup_size = require_list(definition, "workgroup_size", definition_path)
        if len(workgroup_size) != 3:
            raise ValueError(f"{definition_path}: workgroup_size must have 3 values")
        entries.append({
            "id": require_string(route, "id", route_path),
            "op": require_string(definition, "op", definition_path),
            "target": args.target,
            "symbol": require_string(definition, "symbol", definition_path),
            "array_name": f"ggml_hrx_hsaco_{require_string(definition, 'id', definition_path)}_{args.target}",
            "data": hsaco_path.read_bytes(),
            "workgroup_size": [int(value) for value in workgroup_size],
            "threads_per_block": int(launch.get("threads_per_block", workgroup_size[0])),
        })

    if not entries:
        raise ValueError(f"{metadata_path}: no routes selected for target {args.target}")

    write_generated_header(Path(args.out_h))
    write_generated_source(Path(args.out_cpp), Path(args.out_h), entries)


if __name__ == "__main__":
    main()
