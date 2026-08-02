#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from utils import hrx_catalog_emit as catalog_emit
from utils import hrx_route_schema as route_schema
import validate_loom_routes as loom


require_string = catalog_emit.require_string
require_list = catalog_emit.require_list
require_int = catalog_emit.require_int
read_json = catalog_emit.read_json
c_array = catalog_emit.c_array
parse_targets = catalog_emit.parse_targets
c_identifier = catalog_emit.c_identifier


def c_string(value):
    return json.dumps(value, ensure_ascii=True)


def write_generated_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_backend_hrx_loom_source_entry {
    const char * name;
    const unsigned char * data;
    size_t size;
    const char * format;
};

struct ggml_backend_hrx_loom_catalog_entry {
    const char * id;
    const char * op;
    const char * target;
    const char * source_name;
    const unsigned char * source_data;
    size_t source_size;
    const char * source_format;
    const char * symbol;
    const ggml_backend_hrx_loom_source_entry * dependencies;
    size_t dependency_count;
    uint32_t workgroup_size[3];
    uint32_t binding_count;
    uint32_t parameter_count;
    uint32_t constant_byte_length;
};

enum ggml_backend_hrx_loom_storage_transform_kind : uint8_t {
    GGML_BACKEND_HRX_LOOM_STORAGE_ROW_GROUP_FIELD_INTERLEAVE,
};

struct ggml_backend_hrx_loom_storage_transform_entry {
    const char * id;
    const char * target;
    const char * type;
    int64_t shape[4];
    bool contiguous;
    const char * name_prefix;
    const char * name_suffix;
    bool decimal_middle;
    ggml_backend_hrx_loom_storage_transform_kind kind;
    size_t outer_count;
    size_t row_count;
    size_t block_count;
    size_t field_count;
    size_t unit_bytes;
    size_t row_group;
    const uint16_t * field_order;
};

const ggml_backend_hrx_loom_catalog_entry * ggml_backend_hrx_loom_catalog_entries(size_t * count);
const ggml_backend_hrx_loom_storage_transform_entry *
ggml_backend_hrx_loom_storage_transform_entries(size_t * count);
""", encoding="utf-8")


def write_generated_source(path, header_path, entries, storage_transforms):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_name = header_path.name
    chunks = [f"#include \"{header_name}\"\n\n"]
    for entry in entries:
        chunks.append(f"static const unsigned char {entry['array_name']}[] = {{\n")
        chunks.append(c_array(entry["source_data"]))
        chunks.append("\n};\n\n")
        for dependency in entry["dependencies"]:
            chunks.append(f"static const unsigned char {dependency['array_name']}[] = {{\n")
            chunks.append(c_array(dependency["source_data"]))
            chunks.append("\n};\n\n")
        if entry["dependencies"]:
            chunks.append(
                f"static const ggml_backend_hrx_loom_source_entry {entry['dependency_array_name']}[] = {{\n"
            )
            for dependency in entry["dependencies"]:
                chunks.append(
                    "    {\n"
                    f"        /* .name = */ {c_string(dependency['source_name'])},\n"
                    f"        /* .data = */ {dependency['array_name']},\n"
                    f"        /* .size = */ sizeof({dependency['array_name']}),\n"
                    f"        /* .format = */ {c_string(dependency['source_format'])},\n"
                    "    },\n"
                )
            chunks.append("};\n\n")

    chunks.append("static const ggml_backend_hrx_loom_catalog_entry GGML_HRX_LOOM_CATALOG[] = {\n")
    for entry in entries:
        workgroup_size = entry["workgroup_size"]
        dependency_entries = entry["dependency_array_name"] if entry["dependencies"] else "nullptr"
        dependency_count = f"sizeof({entry['dependency_array_name']}) / sizeof({entry['dependency_array_name']}[0])" if entry["dependencies"] else "0"
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
            f"        /* .dependencies = */ {dependency_entries},\n"
            f"        /* .dependency_count = */ {dependency_count},\n"
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
    for index, entry in enumerate(storage_transforms):
        order = ", ".join(str(value) for value in entry["field_order"])
        chunks.append(
            f"\nstatic const uint16_t GGML_HRX_LOOM_STORAGE_FIELD_ORDER_{index}[] = "
            f"{{{order}}};\n"
        )
    if storage_transforms:
        chunks.append(
            "\nstatic const ggml_backend_hrx_loom_storage_transform_entry "
            "GGML_HRX_LOOM_STORAGE_TRANSFORMS[] = {\n"
        )
        for index, entry in enumerate(storage_transforms):
            shape = ", ".join(str(value) for value in entry["shape"])
            chunks.append(
                "    {\n"
                f"        /* .id = */ {c_string(entry['id'])},\n"
                f"        /* .target = */ {c_string(entry['target'])},\n"
                f"        /* .type = */ {c_string(entry['type'])},\n"
                f"        /* .shape = */ {{{shape}}},\n"
                f"        /* .contiguous = */ {'true' if entry['contiguous'] else 'false'},\n"
                f"        /* .name_prefix = */ {c_string(entry['name_prefix'])},\n"
                f"        /* .name_suffix = */ {c_string(entry['name_suffix'])},\n"
                f"        /* .decimal_middle = */ {'true' if entry['decimal_middle'] else 'false'},\n"
                "        /* .kind = */ GGML_BACKEND_HRX_LOOM_STORAGE_ROW_GROUP_FIELD_INTERLEAVE,\n"
                f"        /* .outer_count = */ {entry['outer_count']},\n"
                f"        /* .row_count = */ {entry['row_count']},\n"
                f"        /* .block_count = */ {entry['block_count']},\n"
                f"        /* .field_count = */ {entry['field_count']},\n"
                f"        /* .unit_bytes = */ {entry['unit_bytes']},\n"
                f"        /* .row_group = */ {entry['row_group']},\n"
                f"        /* .field_order = */ GGML_HRX_LOOM_STORAGE_FIELD_ORDER_{index},\n"
                "    },\n"
            )
        chunks.append("};\n\n")
        storage_count = (
            "sizeof(GGML_HRX_LOOM_STORAGE_TRANSFORMS) / "
            "sizeof(GGML_HRX_LOOM_STORAGE_TRANSFORMS[0])"
        )
        storage_pointer = "GGML_HRX_LOOM_STORAGE_TRANSFORMS"
    else:
        storage_count = "0"
        storage_pointer = "nullptr"
    chunks.append("""const ggml_backend_hrx_loom_storage_transform_entry *
ggml_backend_hrx_loom_storage_transform_entries(size_t * count) {
    if (count) {
        *count = %s;
    }
    return %s;
}
""" % (storage_count, storage_pointer))
    path.write_text("".join(chunks), encoding="utf-8")


def catalog_entry_from_definition(source_root, route_id, definition_path, definition, target):
    source_name = require_string(definition, "source", definition_path)
    source_path = (source_root / source_name).resolve()
    if not source_path.is_file():
        raise ValueError(f"{definition_path}: missing source {source_path}")
    source_format = require_string(definition, "source_format", definition_path)
    if source_format not in loom.SOURCE_FORMATS:
        supported = ", ".join(sorted(loom.SOURCE_FORMATS))
        raise ValueError(f"{definition_path}: unsupported source_format {source_format}; expected one of {supported}")

    definition_id = require_string(definition, "id", definition_path)
    dependencies = []
    for i, dependency in enumerate(definition.get("dependencies", [])):
        source = f"{definition_path}: dependencies[{i}]"
        if not isinstance(dependency, dict):
            raise ValueError(f"{source}: expected object")
        dependency_source_name = require_string(dependency, "source", source)
        dependency_source_path = (source_root / dependency_source_name).resolve()
        if not dependency_source_path.is_file():
            raise ValueError(f"{source}: missing source {dependency_source_path}")
        dependency_source_format = require_string(dependency, "source_format", source)
        if dependency_source_format not in loom.DEPENDENCY_SOURCE_FORMATS:
            supported = ", ".join(sorted(loom.DEPENDENCY_SOURCE_FORMATS))
            raise ValueError(
                f"{source}: unsupported source_format {dependency_source_format}; expected one of {supported}"
            )
        dependencies.append({
            "source_name": dependency_source_name,
            "source_data": dependency_source_path.read_bytes(),
            "source_format": dependency_source_format,
            "array_name": (
                f"ggml_hrx_loom_{c_identifier(definition_id)}_"
                f"{c_identifier(route_id)}_{c_identifier(target)}_dep_{i}"
            ),
        })

    workgroup_size = require_list(definition, "workgroup_size", definition_path)
    if len(workgroup_size) != 3:
        raise ValueError(f"{definition_path}: workgroup_size must have 3 values")
    abi = definition.get("abi")
    if not isinstance(abi, dict):
        raise ValueError(f"{definition_path}: expected object field abi")

    return {
        "id": route_id,
        "op": require_string(definition, "op", definition_path),
        "target": target,
        "source_name": source_name,
        "source_data": source_path.read_bytes(),
        "source_format": source_format,
        "symbol": require_string(definition, "symbol", definition_path),
        "array_name": (
            f"ggml_hrx_loom_{c_identifier(definition_id)}_"
            f"{c_identifier(route_id)}_{c_identifier(target)}"
        ),
        "dependencies": dependencies,
        "dependency_array_name": (
            f"ggml_hrx_loom_{c_identifier(definition_id)}_"
            f"{c_identifier(route_id)}_{c_identifier(target)}_deps"
        ),
        "workgroup_size": [int(value) for value in workgroup_size],
        "binding_count": require_int(abi, "binding_count", definition_path),
        "parameter_count": require_int(abi, "parameter_count", definition_path),
        "constant_byte_length": require_int(abi, "constant_byte_length", definition_path),
    }


def route_catalog_entries(source_root, route_path, route, definitions, target):
    route_id = require_string(route, "id", route_path)
    dispatches = loom.route_dispatches(route, route_path)
    dispatch_definitions = loom.resolve_dispatch_definitions(route, route_path, definitions)
    if len(dispatches) == 1:
        definition_path, definition = dispatch_definitions[0]
        return [catalog_entry_from_definition(source_root, route_id, definition_path, definition, target)]

    entries = []
    for dispatch, (definition_path, definition) in zip(dispatches, dispatch_definitions):
        dispatch_name = route_schema.require_string(dispatch, "name", route_path)
        entries.append(catalog_entry_from_definition(source_root, f"{route_id}::{dispatch_name}", definition_path, definition, target))
    return entries


def build_entries(source_root, targets):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    metadata_targets = loom.metadata_targets(metadata_path, metadata)
    selected_targets = targets
    if not selected_targets:
        selected_targets = require_list(metadata, "targets", metadata_path)
    selected_target_names = {}
    for target in selected_targets:
        if not isinstance(target, str) or not target:
            raise ValueError(f"{metadata_path}: targets must contain non-empty strings")
        if target in selected_target_names:
            raise ValueError(f"{metadata_path}: duplicate selected target {target}")
        if target not in metadata_targets:
            raise ValueError(f"{metadata_path}: selected target {target} is not listed in metadata targets")
        selected_target_names[target] = True
    selected_target_entries = {target: 0 for target in selected_targets}

    entries = []
    definitions = loom.load_definitions(source_root)
    for route_name in require_list(metadata, "routes", metadata_path):
        if not isinstance(route_name, str) or not route_name:
            raise ValueError(f"{metadata_path}: routes must contain non-empty strings")
        route_path = source_root / route_name
        route = read_json(route_path)
        route_id = require_string(route, "id", route_path)
        route_architectures = set(require_list(route, "architectures", route_path))
        for target in selected_targets:
            if loom.ARCHITECTURE_ANY not in route_architectures and target not in route_architectures:
                continue
            selected_target_entries[target] += 1
            entries.extend(route_catalog_entries(source_root, route_path, route, definitions, target))

    if not entries:
        raise ValueError(f"{metadata_path}: no routes selected for targets {selected_targets}")
    for target, entry_count in selected_target_entries.items():
        if entry_count == 0:
            raise ValueError(f"{metadata_path}: no Loom routes support selected target {target}")
    return entries


def require_positive_int(data, key, source):
    value = require_int(data, key, source)
    if value <= 0:
        raise ValueError(f"{source}: {key} must be positive")
    return value


def build_storage_transforms(source_root, targets):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    selected_targets = targets or require_list(metadata, "targets", metadata_path)
    result = []
    ids = set()
    for transform_name in metadata.get("storage_transforms", []):
        if not isinstance(transform_name, str) or not transform_name:
            raise ValueError(f"{metadata_path}: storage_transforms must contain non-empty strings")
        path = (source_root / transform_name).resolve()
        value = read_json(path)
        if value.get("schema") != "ggml-hrx-loom-storage-transform-v1":
            raise ValueError(f"{path}: unsupported storage transform schema")
        transform_id = require_string(value, "id", path)
        if transform_id in ids:
            raise ValueError(f"{path}: duplicate storage transform id {transform_id}")
        ids.add(transform_id)
        architectures = set(require_list(value, "architectures", path))
        match = value.get("match")
        transform = value.get("transform")
        if not isinstance(match, dict) or not isinstance(transform, dict):
            raise ValueError(f"{path}: match and transform must be objects")
        tensor_type = require_string(match, "type", path)
        if tensor_type not in loom.route_schema.DTYPES:
            raise ValueError(f"{path}: unsupported tensor type {tensor_type}")
        shape = require_list(match, "shape", path)
        if len(shape) != 4 or any(type(item) is not int or item <= 0 for item in shape):
            raise ValueError(f"{path}: match.shape must contain four positive integers")
        contiguous = match.get("contiguous")
        if type(contiguous) is not bool:
            raise ValueError(f"{path}: match.contiguous must be boolean")
        name = match.get("name")
        if not isinstance(name, dict):
            raise ValueError(f"{path}: match.name must be an object")
        name_prefix = require_string(name, "prefix", path)
        name_suffix = require_string(name, "suffix", path)
        middle = require_string(name, "middle", path)
        if middle != "decimal":
            raise ValueError(f"{path}: only decimal name middles are supported")
        if require_string(transform, "kind", path) != "row_group_field_interleave":
            raise ValueError(f"{path}: unsupported storage transform kind")
        outer_count = require_positive_int(transform, "outer_count", path)
        row_count = require_positive_int(transform, "row_count", path)
        block_count = require_positive_int(transform, "block_count", path)
        field_count = require_positive_int(transform, "field_count", path)
        unit_bytes = require_positive_int(transform, "unit_bytes", path)
        row_group = require_positive_int(transform, "row_group", path)
        if row_count % row_group != 0:
            raise ValueError(f"{path}: row_count must be divisible by row_group")
        field_order = require_list(transform, "field_order", path)
        if (len(field_order) != field_count or
                sorted(field_order) != list(range(field_count))):
            raise ValueError(f"{path}: field_order must be a permutation of all fields")
        for target in selected_targets:
            if loom.ARCHITECTURE_ANY not in architectures and target not in architectures:
                continue
            result.append({
                "id": transform_id,
                "target": target,
                "type": tensor_type,
                "shape": shape,
                "contiguous": contiguous,
                "name_prefix": name_prefix,
                "name_suffix": name_suffix,
                "decimal_middle": True,
                "outer_count": outer_count,
                "row_count": row_count,
                "block_count": block_count,
                "field_count": field_count,
                "unit_bytes": unit_bytes,
                "row_group": row_group,
                "field_order": field_order,
            })
    return result


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
        targets = parse_targets(args.targets)
        entries = build_entries(source_root, targets)
        storage_transforms = build_storage_transforms(source_root, targets)
        write_generated_header(Path(args.out_h))
        write_generated_source(
            Path(args.out_cpp), Path(args.out_h), entries, storage_transforms)
    except (OSError, json.JSONDecodeError, ValueError) as err:
        print(f"ValueError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
