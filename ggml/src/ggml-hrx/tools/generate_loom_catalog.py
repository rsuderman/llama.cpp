#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import validate_loom_routes as loom


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


def require_int(data, key, source):
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"{source}: expected integer field {key}")
    return value


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def c_array(data):
    rows = []
    for i in range(0, len(data), 12):
        rows.append("    " + ", ".join(f"0x{byte:02x}" for byte in data[i:i + 12]))
    return ",\n".join(rows)


def c_string(value):
    return json.dumps(value, ensure_ascii=True)


def parse_targets(value):
    if value is None:
        return []
    return [target for target in re.split(r"[;,\s]+", value) if target]


def c_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def write_generated_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_backend_hrx_loom_catalog_entry {
    const char * id;
    const char * op;
    const char * target;
    const char * source_name;
    const unsigned char * source_data;
    size_t source_size;
    const char * source_format;
    const char * symbol;
    uint32_t workgroup_size[3];
    uint32_t binding_count;
    uint32_t parameter_count;
    uint32_t constant_byte_length;
};

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_catalog_entries(size_t * count);
""", encoding="utf-8")


def write_generated_source(path, header_path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_name = header_path.name
    chunks = [f"#include \"{header_name}\"\n\n"]
    for entry in entries:
        chunks.append(f"static const unsigned char {entry['array_name']}[] = {{\n")
        chunks.append(c_array(entry["source_data"]))
        chunks.append("\n};\n\n")

    chunks.append("static const ggml_backend_hrx_loom_catalog_entry GGML_HRX_LOOM_CATALOG[] = {\n")
    for entry in entries:
        workgroup_size = entry["workgroup_size"]
        chunks.append(
            "    {\n"
            f"        /* .id = */ {c_string(entry['id'])},\n"
            f"        /* .op = */ {c_string(entry['op'])},\n"
            f"        /* .target = */ {c_string(entry['target'])},\n"
            f"        /* .source_name = */ {c_string(entry['source_name'])},\n"
            f"        /* .source_data = */ {entry['array_name']},\n"
            f"        /* .source_size = */ sizeof({entry['array_name']}),\n"
            f"        /* .source_format = */ {c_string(entry['source_format'])},\n"
            f"        /* .symbol = */ {c_string(entry['symbol'])},\n"
            f"        /* .workgroup_size = */ {{{workgroup_size[0]}, {workgroup_size[1]}, {workgroup_size[2]}}},\n"
            f"        /* .binding_count = */ {entry['binding_count']},\n"
            f"        /* .parameter_count = */ {entry['parameter_count']},\n"
            f"        /* .constant_byte_length = */ {entry['constant_byte_length']},\n"
            "    },\n"
        )
    chunks.append("};\n\n")
    chunks.append("""const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_catalog_entries(size_t * count) {
    if (count) {
        *count = sizeof(GGML_HRX_LOOM_CATALOG) / sizeof(GGML_HRX_LOOM_CATALOG[0]);
    }
    return GGML_HRX_LOOM_CATALOG;
}
""")
    path.write_text("".join(chunks), encoding="utf-8")


def build_entries(source_root, targets):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    selected_targets = targets
    if not selected_targets:
        selected_targets = require_list(metadata, "targets", metadata_path)
    for target in selected_targets:
        if not isinstance(target, str) or not target:
            raise ValueError(f"{metadata_path}: targets must contain non-empty strings")

    entries = []
    for route_name in require_list(metadata, "routes", metadata_path):
        if not isinstance(route_name, str) or not route_name:
            raise ValueError(f"{metadata_path}: routes must contain non-empty strings")
        route_path = source_root / route_name
        route = read_json(route_path)
        definition_path = (route_path.parent / require_string(route, "definition", route_path)).resolve()
        definition = read_json(definition_path)
        source_name = require_string(definition, "source", definition_path)
        source_path = (source_root / source_name).resolve()
        if not source_path.is_file():
            raise ValueError(f"{definition_path}: missing source {source_path}")
        source_format = require_string(definition, "source_format", definition_path)
        if source_format not in loom.SOURCE_FORMATS:
            supported = ", ".join(sorted(loom.SOURCE_FORMATS))
            raise ValueError(f"{definition_path}: unsupported source_format {source_format}; expected one of {supported}")
        workgroup_size = require_list(definition, "workgroup_size", definition_path)
        if len(workgroup_size) != 3:
            raise ValueError(f"{definition_path}: workgroup_size must have 3 values")
        abi = definition.get("abi")
        if not isinstance(abi, dict):
            raise ValueError(f"{definition_path}: expected object field abi")

        route_id = require_string(route, "id", route_path)
        definition_id = require_string(definition, "id", definition_path)
        source_data = source_path.read_bytes()
        for target in selected_targets:
            entries.append({
                "id": route_id,
                "op": require_string(definition, "op", definition_path),
                "target": target,
                "source_name": source_name,
                "source_data": source_data,
                "source_format": source_format,
                "symbol": require_string(definition, "symbol", definition_path),
                "array_name": (
                    f"ggml_hrx_loom_{c_identifier(definition_id)}_"
                    f"{c_identifier(route_id)}_{c_identifier(target)}"
                ),
                "workgroup_size": [int(value) for value in workgroup_size],
                "binding_count": require_int(abi, "binding_count", definition_path),
                "parameter_count": require_int(abi, "parameter_count", definition_path),
                "constant_byte_length": require_int(abi, "constant_byte_length", definition_path),
            })

    if not entries:
        raise ValueError(f"{metadata_path}: no routes selected for targets {selected_targets}")
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-cpp", required=True)
    parser.add_argument("--out-h", required=True)
    parser.add_argument("--targets")
    args = parser.parse_args()

    try:
        source_root = Path(args.source_root)
        loom.validate_catalog(source_root)
        entries = build_entries(source_root, parse_targets(args.targets))
        write_generated_header(Path(args.out_h))
        write_generated_source(Path(args.out_cpp), Path(args.out_h), entries)
    except (OSError, json.JSONDecodeError, ValueError) as err:
        print(f"ValueError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
