#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

from utils import hrx_route_schema as route_schema


CATALOG_SCHEMA_V0 = "ggml-hrx-loom-catalog-v0"
DEFINITION_SCHEMA_V1 = "ggml-hrx-loom-def-v1"
ROUTE_SCHEMA_V1 = "ggml-hrx-loom-route-v1"
ROUTE_FORMAT = "loom"
ARCHITECTURE_ANY = "*"

SOURCE_FORMATS = {"loom-text", "loom-bytecode", "amdgpu-hsaco"}

METADATA_FIELDS = {"schema", "version", "targets", "routes"}
DEFINITION_FIELDS = {
    "schema",
    "id",
    "op",
    "source",
    "source_format",
    "symbol",
    "abi",
    "workgroup_size",
    "parameters",
    "bindings",
}
TOP_LEVEL_FIELDS = {
    "schema",
    "id",
    "architectures",
    "definition",
    "format",
    "priority",
    "match",
    "derived",
    "config",
    "invocation",
    "tests",
}
CONFIG_FIELDS = {"mode", "bindings"}
CONFIG_BINDING_FIELDS = {"name", "source", "value", "type"}

MAX_CONFIG_BINDINGS = 32
MAX_CONFIG_NAME_BYTES = 63
MAX_CONFIG_VALUE_BYTES = 127

CONFIG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_source_root():
    return Path(__file__).resolve().parents[1] / "loom-catalog"


def validate_metadata(source_root):
    metadata_path = source_root / "metadata.json"
    metadata = route_schema.read_json(metadata_path)
    route_schema.unknown_fields(metadata, METADATA_FIELDS, metadata_path)
    if route_schema.require_string(metadata, "schema", metadata_path) != CATALOG_SCHEMA_V0:
        raise ValueError(f"{metadata_path}: unsupported schema")
    route_schema.require_int(metadata, "version", metadata_path)
    targets = {}
    for target in route_schema.require_list(metadata, "targets", metadata_path):
        if not isinstance(target, str) or not target:
            raise ValueError(f"{metadata_path}: targets must contain non-empty strings")
        if target in targets:
            raise ValueError(f"{metadata_path}: duplicate target {target}")
        targets[target] = True
    return metadata_path, metadata


def validate_definition(definition, definition_path, source_root):
    route_schema.unknown_fields(definition, DEFINITION_FIELDS, definition_path)
    schema = route_schema.require_string(definition, "schema", definition_path)
    if schema != DEFINITION_SCHEMA_V1:
        raise ValueError(f"{definition_path}: unsupported definition schema {schema}")
    source_format = route_schema.require_string(definition, "source_format", definition_path)
    if source_format not in SOURCE_FORMATS:
        supported = ", ".join(sorted(SOURCE_FORMATS))
        raise ValueError(f"{definition_path}: unsupported source_format {source_format}; expected one of {supported}")
    route_schema.validate_definition(definition, definition_path, source_root)


def load_definitions(source_root):
    definitions = {}
    definition_ids = {}
    definitions_dir = source_root / "defs"
    if not definitions_dir.is_dir():
        raise ValueError(f"{source_root}: missing defs directory")
    for definition_path in sorted(definitions_dir.rglob("*.json")):
        definition = route_schema.read_json(definition_path)
        validate_definition(definition, definition_path, source_root)
        definition_id = route_schema.require_string(definition, "id", definition_path)
        existing = definition_ids.get(definition_id)
        if existing is not None:
            raise ValueError(f"{definition_path}: duplicate definition id {definition_id} also used by {existing}")
        definition_ids[definition_id] = definition_path
        definitions[definition_path.resolve()] = definition
    return definitions


def metadata_targets(metadata_path, metadata):
    return set(route_schema.require_list(metadata, "targets", metadata_path))


def load_metadata_targets(source_root):
    metadata_path, metadata = validate_metadata(source_root)
    return metadata_targets(metadata_path, metadata)


def validate_config(route, route_path, context):
    config = route_schema.require_dict(route, "config", route_path)
    route_schema.unknown_fields(config, CONFIG_FIELDS, f"{route_path}: config")
    mode = route_schema.require_string(config, "mode", f"{route_path}: config")
    if mode != "compile":
        raise ValueError(f"{route_path}: config.mode must be compile")
    bindings = route_schema.require_array(config, "bindings", f"{route_path}: config")
    if len(bindings) > MAX_CONFIG_BINDINGS:
        raise ValueError(f"{route_path}: config.bindings must have at most {MAX_CONFIG_BINDINGS} entries")

    names = {}
    for i, binding in enumerate(bindings):
        source = f"{route_path}: config.bindings[{i}]"
        if not isinstance(binding, dict):
            raise ValueError(f"{source}: expected object")
        route_schema.unknown_fields(binding, CONFIG_BINDING_FIELDS, source)

        name = route_schema.require_string(binding, "name", source)
        if CONFIG_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"{source}: name must match [A-Za-z_][A-Za-z0-9_]*")
        if len(name.encode("utf-8")) > MAX_CONFIG_NAME_BYTES:
            raise ValueError(f"{source}: name must fit in 63 bytes")
        if name in names:
            raise ValueError(f"{route_path}: duplicate config binding name {name}")
        names[name] = source

        binding_type = route_schema.require_string(binding, "type", source)
        if binding_type not in route_schema.SCALAR_TYPES:
            supported = ", ".join(sorted(route_schema.SCALAR_TYPES))
            raise ValueError(f"{source}: unsupported config binding type {binding_type}; expected one of {supported}")

        has_source = "source" in binding
        has_value = "value" in binding
        if has_source == has_value:
            raise ValueError(f"{source}: expected exactly one of source or value")
        if has_source:
            value_type = context.resolve_source(route_schema.require_string(binding, "source", source), f"{source}.source")
            if value_type != binding_type:
                raise ValueError(f"{source}.source: source type {value_type} does not match config type {binding_type}")
        else:
            route_schema.validate_literal(binding["value"], binding_type, f"{source}.value")
            value_text = str(binding["value"])
            if len(value_text.encode("utf-8")) > MAX_CONFIG_VALUE_BYTES:
                raise ValueError(f"{source}.value: string representation must fit in 127 bytes")


def validate_architectures(route, route_path, metadata_targets):
    architectures = route_schema.require_array(route, "architectures", route_path)
    if not architectures:
        raise ValueError(f"{route_path}: architectures must not be empty")

    names = {}
    for i, architecture in enumerate(architectures):
        source = f"{route_path}: architectures[{i}]"
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"{source}: expected non-empty string")
        if architecture in names:
            raise ValueError(f"{route_path}: duplicate architecture {architecture}")
        if architecture != ARCHITECTURE_ANY and architecture not in metadata_targets:
            raise ValueError(f"{source}: architecture {architecture} is not listed in metadata targets")
        names[architecture] = True
    if ARCHITECTURE_ANY in names:
        if len(names) != 1:
            raise ValueError(f"{route_path}: architecture {ARCHITECTURE_ANY} cannot be combined with explicit architectures")
        return set(metadata_targets)
    return set(names)


def validate_route(route_path, definitions, metadata_targets):
    route = route_schema.read_json(route_path)
    route_schema.unknown_fields(route, TOP_LEVEL_FIELDS, route_path)
    schema = route_schema.require_string(route, "schema", route_path)
    if schema != ROUTE_SCHEMA_V1:
        raise ValueError(f"{route_path}: unsupported route schema {schema}")
    route_schema.require_string(route, "id", route_path)
    architectures = validate_architectures(route, route_path, metadata_targets)
    route_format = route_schema.require_string(route, "format", route_path)
    if route_format != ROUTE_FORMAT:
        raise ValueError(f"{route_path}: unsupported route format {route_format}")
    route_schema.require_int(route, "priority", route_path)
    if "tests" in route:
        route_schema.require_dict(route, "tests", route_path)

    definition_path = (route_path.parent / route_schema.require_string(route, "definition", route_path)).resolve()
    definition = definitions.get(definition_path)
    if definition is None:
        raise ValueError(f"{route_path}: missing definition {definition_path}")

    op_rule, tensors, attributes, predicates = route_schema.validate_match(route, route_path, definition)
    derived = route_schema.require_dict(route, "derived", route_path)
    context = route_schema.RouteContext(route_path, op_rule, tensors, attributes, derived)
    route_schema.validate_derived(route, route_path, context)
    route_schema.validate_predicates(predicates, route_path, context)
    validate_config(route, route_path, context)
    route_schema.validate_invocation(route, route_path, definition, context)
    return architectures


def validate_catalog(source_root):
    metadata_path, metadata = validate_metadata(source_root)
    targets = metadata_targets(metadata_path, metadata)
    definitions = load_definitions(source_root)
    route_names = route_schema.require_list(metadata, "routes", metadata_path)
    covered_targets = set()
    for route_name in route_names:
        if not isinstance(route_name, str) or not route_name:
            raise ValueError(f"{metadata_path}: routes must contain non-empty strings")
        route_path = source_root / route_name
        if not route_path.is_file():
            raise ValueError(f"{metadata_path}: missing route {route_path}")
        covered_targets.update(validate_route(route_path, definitions, targets))
    for target in sorted(targets - covered_targets):
        raise ValueError(f"{metadata_path}: metadata target {target} is not covered by any route architecture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(default_source_root()))
    args = parser.parse_args()

    try:
        validate_catalog(Path(args.source_root))
    except (OSError, json.JSONDecodeError, ValueError) as err:
        print(f"ValueError: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
