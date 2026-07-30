#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from utils import hrx_route_schema as route_schema


CATALOG_SCHEMA_V0 = "ggml-hrx-hsaco-catalog-v0"
ROUTE_SCHEMA_V1 = "ggml-hrx-hsaco-route-v1"
ROUTE_FORMAT = "amdgpu-hsaco"

TOP_LEVEL_FIELDS = {
    "schema",
    "id",
    "definition",
    "format",
    "priority",
    "match",
    "derived",
    "invocation",
    "tests",
    "routing",
}

MATCH_FIELDS = route_schema.MATCH_FIELDS
TENSOR_FIELDS = route_schema.TENSOR_FIELDS
ATTRIBUTE_FIELDS = route_schema.ATTRIBUTE_FIELDS
PREDICATE_FIELDS = route_schema.PREDICATE_FIELDS
DERIVED_FIELDS = route_schema.DERIVED_FIELDS
BUFFER_FIELDS = route_schema.BUFFER_FIELDS
SCALAR_FIELDS = route_schema.SCALAR_FIELDS
DISPATCH_FIELDS = route_schema.DISPATCH_FIELDS
INVOCATION_FIELDS = route_schema.INVOCATION_FIELDS

SCALAR_TYPES = route_schema.SCALAR_TYPES
INTEGER_TYPES = route_schema.INTEGER_TYPES
DTYPES = route_schema.DTYPES
SUPPORTED_BINDING_ACCESSES = route_schema.SUPPORTED_BINDING_ACCESSES
SUPPORTED_SCALAR_TYPES = route_schema.SUPPORTED_SCALAR_TYPES
OP_RULES = route_schema.OP_RULES
BUFFER_KIND_ACCESS = route_schema.BUFFER_KIND_ACCESS

RouteContext = route_schema.RouteContext
align_offset = route_schema.align_offset
is_source_string = route_schema.is_source_string
packed_parameter_size = route_schema.packed_parameter_size
read_json = route_schema.read_json
require_array = route_schema.require_array
require_bool = route_schema.require_bool
require_dict = route_schema.require_dict
require_int = route_schema.require_int
require_list = route_schema.require_list
require_non_empty_dict = route_schema.require_non_empty_dict
require_string = route_schema.require_string
route_tensors = route_schema.route_tensors
unknown_fields = route_schema.unknown_fields
validate_attributes = route_schema.validate_attributes
validate_buffers = route_schema.validate_buffers
validate_ceil_div = route_schema.validate_ceil_div
validate_comparator_value = route_schema.validate_comparator_value
validate_definition = route_schema.validate_definition
validate_derived = route_schema.validate_derived
validate_dispatch = route_schema.validate_dispatch
validate_field_predicate = route_schema.validate_field_predicate
validate_integer_operand = route_schema.validate_integer_operand
validate_integer_source = route_schema.validate_integer_source
validate_invocation = route_schema.validate_invocation
validate_literal = route_schema.validate_literal
validate_match = route_schema.validate_match
validate_positions = route_schema.validate_positions
validate_predicates = route_schema.validate_predicates
validate_product = route_schema.validate_product
validate_same_shape_predicate = route_schema.validate_same_shape_predicate
validate_scalars = route_schema.validate_scalars
validate_tensors = route_schema.validate_tensors
validate_unique_names = route_schema.validate_unique_names
validate_workgroup_size = route_schema.validate_workgroup_size


def default_source_root():
    return Path(__file__).resolve().parents[1] / "hsaco-catalog"


def validate_metadata(source_root):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    if require_string(metadata, "schema", metadata_path) != CATALOG_SCHEMA_V0:
        raise ValueError(f"{metadata_path}: unsupported schema")
    for target in require_list(metadata, "targets", metadata_path):
        if not isinstance(target, str) or not target:
            raise ValueError(f"{metadata_path}: targets must contain non-empty strings")
    return metadata_path, metadata


def load_definitions(source_root):
    definitions = {}
    definition_ids = {}
    definitions_dir = source_root / "defs"
    if not definitions_dir.is_dir():
        raise ValueError(f"{source_root}: missing defs directory")
    for definition_path in sorted(definitions_dir.glob("*.json")):
        definition = read_json(definition_path)
        validate_definition(definition, definition_path, source_root)
        definition_id = require_string(definition, "id", definition_path)
        existing = definition_ids.get(definition_id)
        if existing is not None:
            raise ValueError(f"{definition_path}: duplicate definition id {definition_id} also used by {existing}")
        definition_ids[definition_id] = definition_path
        definitions[definition_path.resolve()] = definition
    return definitions


def validate_route(route_path, definitions):
    route = read_json(route_path)
    unknown_fields(route, TOP_LEVEL_FIELDS, route_path)
    schema = require_string(route, "schema", route_path)
    if schema != ROUTE_SCHEMA_V1:
        raise ValueError(f"{route_path}: unsupported route schema {schema}")
    require_string(route, "id", route_path)
    route_format = require_string(route, "format", route_path)
    if route_format != ROUTE_FORMAT:
        raise ValueError(f"{route_path}: unsupported route format {route_format}")
    require_int(route, "priority", route_path)
    if "tests" in route:
        require_dict(route, "tests", route_path)
    if "routing" in route:
        require_dict(route, "routing", route_path)

    definition_path = (route_path.parent / require_string(route, "definition", route_path)).resolve()
    definition = definitions.get(definition_path)
    if definition is None:
        raise ValueError(f"{route_path}: missing definition {definition_path}")

    op_rule, tensors, attributes, predicates = validate_match(route, route_path, definition)
    derived = require_dict(route, "derived", route_path)
    context = RouteContext(route_path, op_rule, tensors, attributes, derived)
    validate_derived(route, route_path, context)
    validate_predicates(predicates, route_path, context)
    validate_invocation(route, route_path, definition, context)


def validate_catalog(source_root):
    metadata_path, metadata = validate_metadata(source_root)
    definitions = load_definitions(source_root)
    route_names = require_list(metadata, "routes", metadata_path)
    for route_name in route_names:
        if not isinstance(route_name, str) or not route_name:
            raise ValueError(f"{metadata_path}: routes must contain non-empty strings")
        route_path = source_root / route_name
        if not route_path.is_file():
            raise ValueError(f"{metadata_path}: missing route {route_path}")
        validate_route(route_path, definitions)


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
