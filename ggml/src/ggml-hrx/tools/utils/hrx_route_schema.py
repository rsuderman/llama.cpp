import json


MATCH_FIELDS = {"op", "tensors", "attributes", "predicates"}
TENSOR_FIELDS = {"type", "optional", "shape"}
ATTRIBUTE_FIELDS = {"type", "default"}
PREDICATE_FIELDS = {"contiguous", "same_shape", "rank", "field", "equals", "in", "min", "max", "multiple_of", "src_absent", "src_present"}
DERIVED_FIELDS = {"type", "field", "value", "product", "ceil_div", "next_power_of_2"}
BUFFER_FIELDS = {"name", "tensor", "position", "kind"}
SCALAR_FIELDS = {"name", "source", "value", "type", "position"}
DISPATCH_FIELDS = {"kind", "work_items", "rows", "workgroup_count", "workgroup_size"}
INVOCATION_FIELDS = {"buffers", "scalars", "dispatch"}
WORKGROUP_COUNT_FIELDS = {"x", "y", "z"}

SCALAR_TYPES = {"i32", "i64", "f32", "f64"}
INTEGER_TYPES = {"i32", "i64"}
DTYPES = {"F32", "I32"}
SUPPORTED_BINDING_ACCESSES = {"read", "write", "read_write"}
SUPPORTED_SCALAR_TYPES = {
    "f32": (4, 4),
    "i32": (4, 4),
    "i64": (8, 8),
}

OP_RULES = {
    "GGML_OP_ADD": {
        "required_tensors": {"src0", "src1", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0", "src1"},
        "attributes": {},
    },
    "GGML_OP_ARGSORT": {
        "required_tensors": {"src0", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0"},
        "attributes": {"order": "i32"},
    },
    "GGML_OP_CLAMP": {
        "required_tensors": {"src0", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0"},
        "attributes": {"minimum": "f32", "maximum": "f32"},
    },
    "GGML_OP_DIV": {
        "required_tensors": {"src0", "src1", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0", "src1"},
        "attributes": {},
    },
    "GGML_OP_MUL": {
        "required_tensors": {"src0", "src1", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0", "src1"},
        "attributes": {},
    },
    "GGML_OP_SCALE": {
        "required_tensors": {"src0", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0"},
        "attributes": {"scale": "f32", "bias": "f32"},
    },
    "GGML_OP_SOFT_MAX": {
        "required_tensors": {"src0", "dst"},
        "optional_tensors": {"src1", "src2"},
        "input_tensors": {"src0", "src1", "src2"},
        "attributes": {"scale": "f32", "max_bias": "f32"},
    },
    "GGML_OP_SUM_ROWS": {
        "required_tensors": {"src0", "dst"},
        "optional_tensors": set(),
        "input_tensors": {"src0"},
        "attributes": {},
    },
}

BUFFER_KIND_ACCESS = {
    "input": "read",
    "output": "write",
    "inout": "read_write",
}


def unknown_fields(data, allowed, source):
    for key in data:
        if key not in allowed:
            raise ValueError(f"{source}: unsupported field {key}")


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def require_dict(data, key, source):
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected object field {key}")
    return value


def require_int(data, key, source):
    value = data.get(key)
    if type(value) is not int:
        raise ValueError(f"{source}: expected integer field {key}")
    return value


def require_array(data, key, source):
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{source}: expected array field {key}")
    return value


def require_bool(data, key, source):
    value = data.get(key)
    if type(value) is not bool:
        raise ValueError(f"{source}: expected boolean field {key}")
    return value


def require_non_empty_dict(data, key, source):
    value = require_dict(data, key, source)
    if not value:
        raise ValueError(f"{source}: expected non-empty object field {key}")
    return value


def align_offset(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment


def packed_parameter_size(parameters, definition_path):
    offset = 0
    for i, parameter in enumerate(parameters):
        parameter_source = f"{definition_path}: parameters[{i}]"
        if not isinstance(parameter, dict):
            raise ValueError(f"{parameter_source}: expected object")
        parameter_type = require_string(parameter, "type", parameter_source)
        if parameter_type not in SUPPORTED_SCALAR_TYPES:
            raise ValueError(f"{parameter_source}: unsupported parameter type {parameter_type}")
        size, alignment = SUPPORTED_SCALAR_TYPES[parameter_type]
        offset = align_offset(offset, alignment)
        offset += size
    return offset


def validate_unique_names(items, item_name, definition_path):
    names = {}
    for i, item in enumerate(items):
        item_source = f"{definition_path}: {item_name}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_source}: expected object")
        name = require_string(item, "name", item_source)
        if name in names:
            raise ValueError(f"{definition_path}: duplicate {item_name} name {name}")
        names[name] = item_source


def is_source_string(value):
    return (
        isinstance(value, str)
        and (
            value.startswith("tensor.")
            or value.startswith("attribute.")
            or value.startswith("derived.")
            or value.startswith("shape.")
        )
    )


def validate_literal(value, expected_type, source):
    if expected_type in INTEGER_TYPES:
        if type(value) is not int:
            raise ValueError(f"{source}: expected integer literal for {expected_type}")
        return
    if expected_type in {"f32", "f64"}:
        if type(value) not in {int, float}:
            raise ValueError(f"{source}: expected numeric literal for {expected_type}")
        return
    if expected_type == "dtype":
        if value not in DTYPES:
            raise ValueError(f"{source}: expected tensor dtype")
        return
    raise ValueError(f"{source}: cannot validate literal for {expected_type}")


def validate_workgroup_size(value, source):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{source}: workgroup_size must have 3 values")
    for i, item in enumerate(value):
        if type(item) is not int or item <= 0:
            raise ValueError(f"{source}: workgroup_size[{i}] must be a positive integer")


def validate_positions(items, item_name, source):
    positions = []
    for i, item in enumerate(items):
        item_source = f"{source}: {item_name}[{i}]"
        positions.append(require_int(item, "position", item_source))
    expected = list(range(len(items)))
    if sorted(positions) != expected:
        raise ValueError(f"{source}: {item_name} positions must be unique and contiguous from 0")


def validate_definition(definition, definition_path, source_root):
    require_string(definition, "id", definition_path)
    require_string(definition, "symbol", definition_path)

    source_path = (source_root / require_string(definition, "source", definition_path)).resolve()
    if not source_path.is_file():
        raise ValueError(f"{definition_path}: missing source {source_path}")

    abi = require_dict(definition, "abi", definition_path)
    binding_count = require_int(abi, "binding_count", definition_path)
    parameter_count = require_int(abi, "parameter_count", definition_path)
    constant_byte_length = require_int(abi, "constant_byte_length", definition_path)
    if binding_count < 0:
        raise ValueError(f"{definition_path}: abi.binding_count must be non-negative")
    if parameter_count < 0:
        raise ValueError(f"{definition_path}: abi.parameter_count must be non-negative")
    if constant_byte_length < 0:
        raise ValueError(f"{definition_path}: abi.constant_byte_length must be non-negative")

    parameters = require_array(definition, "parameters", definition_path)
    bindings = require_array(definition, "bindings", definition_path)
    if binding_count != len(bindings):
        raise ValueError(
            f"{definition_path}: abi.binding_count {binding_count} does not match bindings length {len(bindings)}")
    expected_parameter_count = binding_count + len(parameters)
    if parameter_count != expected_parameter_count:
        raise ValueError(
            f"{definition_path}: abi.parameter_count {parameter_count} does not match "
            f"binding count plus parameters length {expected_parameter_count}")

    validate_unique_names(parameters, "parameters", definition_path)
    validate_unique_names(bindings, "bindings", definition_path)

    for i, binding in enumerate(bindings):
        binding_source = f"{definition_path}: bindings[{i}]"
        access = require_string(binding, "access", binding_source)
        if access not in SUPPORTED_BINDING_ACCESSES:
            supported = ", ".join(sorted(SUPPORTED_BINDING_ACCESSES))
            raise ValueError(f"{binding_source}: unsupported access {access}; expected one of {supported}")

    validate_workgroup_size(require_array(definition, "workgroup_size", definition_path), definition_path)

    packed_size = packed_parameter_size(parameters, definition_path)
    if packed_size != constant_byte_length:
        raise ValueError(
            f"{definition_path}: packed parameter size {packed_size} does not match "
            f"abi.constant_byte_length {constant_byte_length}")


class RouteContext:
    def __init__(self, route_path, op_rule, tensors, attributes, derived):
        self.route_path = route_path
        self.op_rule = op_rule
        self.tensors = tensors
        self.attributes = attributes
        self.derived = derived
        self.shape_captures = {}
        for role, tensor in tensors.items():
            self.shape_captures[role] = tensor.get("shape", [])

    def validate_tensor_role(self, role, source, allow_optional=False, require_input=False):
        if not isinstance(role, str) or not role:
            raise ValueError(f"{source}: expected non-empty tensor role")
        known_tensors = self.op_rule["required_tensors"] | self.op_rule["optional_tensors"]
        if role not in known_tensors:
            raise ValueError(f"{source}: unsupported tensor role {role}")
        if require_input and role not in self.op_rule["input_tensors"]:
            raise ValueError(f"{source}: expected source tensor role, got {role}")
        if role not in self.tensors and not allow_optional:
            raise ValueError(f"{source}: tensor role {role} is not declared in match.tensors")

    def resolve_source(self, value, source, derived_names=None):
        if not is_source_string(value):
            raise ValueError(f"{source}: expected source string")
        parts = value.split(".")
        if parts[0] == "tensor":
            return self.resolve_tensor_source(parts, source)
        if parts[0] == "attribute":
            return self.resolve_attribute_source(parts, source)
        if parts[0] == "derived":
            return self.resolve_derived_source(parts, source, derived_names)
        if parts[0] == "shape":
            return self.resolve_shape_source(parts, source)
        raise ValueError(f"{source}: unsupported source {value}")

    def resolve_tensor_source(self, parts, source):
        if len(parts) not in {3, 4}:
            raise ValueError(f"{source}: unsupported tensor source")
        role = parts[1]
        self.validate_tensor_role(role, source)
        field = parts[2]
        if field == "type" and len(parts) == 3:
            return "dtype"
        if field == "rank" and len(parts) == 3:
            return "i64"
        if field == "element_count" and len(parts) == 3:
            return "i64"
        if field in {"dimensions", "strides"}:
            if len(parts) == 3:
                return "vector_i64"
            index = parts[3]
            if not index.isdigit():
                raise ValueError(f"{source}: tensor {field} index must be a non-negative integer")
            return "i64"
        raise ValueError(f"{source}: unsupported tensor source")

    def resolve_attribute_source(self, parts, source):
        if len(parts) != 2:
            raise ValueError(f"{source}: unsupported attribute source")
        name = parts[1]
        attr = self.attributes.get(name)
        if attr is None:
            raise ValueError(f"{source}: attribute {name} is not declared in match.attributes")
        return attr["type"]

    def resolve_derived_source(self, parts, source, derived_names):
        if len(parts) != 2:
            raise ValueError(f"{source}: unsupported derived source")
        name = parts[1]
        names = self.derived if derived_names is None else derived_names
        if name not in names:
            raise ValueError(f"{source}: derived value {name} is not available")
        return self.derived[name]["type"]

    def resolve_shape_source(self, parts, source):
        if len(parts) != 3:
            raise ValueError(f"{source}: unsupported shape source")
        role = parts[1]
        name = parts[2]
        self.validate_tensor_role(role, source)
        captures = self.shape_captures.get(role, [])
        if name not in captures:
            raise ValueError(f"{source}: shape value {name} is not captured for tensor {role}")
        return "i64"


def validate_match(route, route_path, definition):
    match = require_dict(route, "match", route_path)
    unknown_fields(match, MATCH_FIELDS, f"{route_path}: match")
    op = require_string(match, "op", f"{route_path}: match")
    if op not in OP_RULES:
        raise ValueError(f"{route_path}: unsupported match.op {op}")
    definition_op = require_string(definition, "op", route_path)
    if op != definition_op:
        raise ValueError(f"{route_path}: match.op {op} does not match definition op {definition_op}")

    op_rule = OP_RULES[op]
    tensors = validate_tensors(match, route_path, op_rule)
    attributes = validate_attributes(match, route_path, op_rule)
    predicates = match.get("predicates", [])
    if not isinstance(predicates, list):
        raise ValueError(f"{route_path}: match.predicates must be an array")
    return op_rule, tensors, attributes, predicates


def validate_tensors(match, route_path, op_rule):
    tensors = require_non_empty_dict(match, "tensors", f"{route_path}: match")
    known_tensors = op_rule["required_tensors"] | op_rule["optional_tensors"]
    missing = sorted(op_rule["required_tensors"] - set(tensors))
    if missing:
        raise ValueError(f"{route_path}: match.tensors missing required roles {', '.join(missing)}")

    for role, tensor in tensors.items():
        tensor_source = f"{route_path}: match.tensors.{role}"
        if role not in known_tensors:
            raise ValueError(f"{tensor_source}: unsupported tensor role")
        if not isinstance(tensor, dict):
            raise ValueError(f"{tensor_source}: expected object")
        unknown_fields(tensor, TENSOR_FIELDS, tensor_source)
        tensor_type = require_string(tensor, "type", tensor_source)
        if tensor_type not in DTYPES:
            raise ValueError(f"{tensor_source}: unsupported tensor type {tensor_type}")
        if "optional" in tensor:
            require_bool(tensor, "optional", tensor_source)
            if role not in op_rule["optional_tensors"]:
                raise ValueError(f"{tensor_source}: optional is only supported for optional operation inputs")
        if "shape" in tensor:
            shape = tensor.get("shape")
            if not isinstance(shape, list):
                raise ValueError(f"{tensor_source}.shape: expected array")
            if len(shape) > 4:
                raise ValueError(f"{tensor_source}.shape: expected at most 4 names")
            seen = set()
            for i, name in enumerate(shape):
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{tensor_source}.shape[{i}]: expected non-empty string")
                if name in seen:
                    raise ValueError(f"{tensor_source}.shape[{i}]: duplicate shape capture name")
                seen.add(name)
    return tensors


def validate_attributes(match, route_path, op_rule):
    attributes = match.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError(f"{route_path}: match.attributes must be an object")
    for name, attribute in attributes.items():
        attr_source = f"{route_path}: match.attributes.{name}"
        expected_type = op_rule["attributes"].get(name)
        if expected_type is None:
            raise ValueError(f"{attr_source}: unsupported attribute for operation")
        if not isinstance(attribute, dict):
            raise ValueError(f"{attr_source}: expected object")
        unknown_fields(attribute, ATTRIBUTE_FIELDS, attr_source)
        attr_type = require_string(attribute, "type", attr_source)
        if attr_type not in SCALAR_TYPES:
            raise ValueError(f"{attr_source}: unsupported attribute type {attr_type}")
        if attr_type != expected_type:
            raise ValueError(f"{attr_source}: type {attr_type} does not match operation type {expected_type}")
        if "default" in attribute:
            validate_literal(attribute["default"], attr_type, f"{attr_source}.default")
    return attributes


def validate_derived(route, route_path, context):
    derived = require_dict(route, "derived", route_path)
    available = set()
    for name, item in derived.items():
        item_source = f"{route_path}: derived.{name}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"{route_path}: derived names must be non-empty strings")
        if not isinstance(item, dict):
            raise ValueError(f"{item_source}: expected object")
        unknown_fields(item, DERIVED_FIELDS, item_source)
        item_type = require_string(item, "type", item_source)
        if item_type not in SCALAR_TYPES:
            raise ValueError(f"{item_source}: unsupported derived type {item_type}")
        ops = [key for key in ("field", "value", "product", "ceil_div", "next_power_of_2") if key in item]
        if len(ops) != 1:
            raise ValueError(f"{item_source}: expected exactly one derived operation")
        op = ops[0]
        if op == "field":
            context.resolve_source(require_string(item, "field", item_source), f"{item_source}.field", available)
        elif op == "value":
            validate_literal(item["value"], item_type, f"{item_source}.value")
        elif op == "product":
            validate_product(item["product"], item_type, context, available, f"{item_source}.product")
        elif op == "ceil_div":
            validate_ceil_div(item["ceil_div"], item_type, context, available, f"{item_source}.ceil_div")
        elif op == "next_power_of_2":
            source_type = context.resolve_source(
                require_string(item, "next_power_of_2", item_source),
                f"{item_source}.next_power_of_2",
                available)
            if source_type not in INTEGER_TYPES:
                raise ValueError(f"{item_source}.next_power_of_2: expected integer source")
        available.add(name)


def validate_product(value, item_type, context, available, source):
    if item_type not in INTEGER_TYPES:
        raise ValueError(f"{source}: product result must be an integer type")
    if is_source_string(value):
        operand_type = context.resolve_source(value, source, available)
        if operand_type != "vector_i64":
            raise ValueError(f"{source}: expected integer vector source")
        return
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: expected non-empty array or integer vector source")
    for i, operand in enumerate(value):
        operand_type = context.resolve_source(operand, f"{source}[{i}]", available)
        if operand_type not in INTEGER_TYPES and operand_type != "vector_i64":
            raise ValueError(f"{source}[{i}]: expected integer or integer vector source")


def validate_ceil_div(value, item_type, context, available, source):
    if item_type not in INTEGER_TYPES:
        raise ValueError(f"{source}: ceil_div result must be an integer type")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{source}: expected [numerator, denominator]")
    for i, operand in enumerate(value):
        if is_source_string(operand):
            operand_type = context.resolve_source(operand, f"{source}[{i}]", available)
            if operand_type not in INTEGER_TYPES:
                raise ValueError(f"{source}[{i}]: expected integer source")
        elif type(operand) is int:
            if i == 1 and operand <= 0:
                raise ValueError(f"{source}[{i}]: denominator must be positive")
        else:
            raise ValueError(f"{source}[{i}]: expected integer literal or source string")


def validate_predicates(predicates, route_path, context):
    for i, predicate in enumerate(predicates):
        source = f"{route_path}: match.predicates[{i}]"
        if not isinstance(predicate, dict):
            raise ValueError(f"{source}: expected object")
        unknown_fields(predicate, PREDICATE_FIELDS, source)
        keys = set(predicate)
        forms = [key for key in ("contiguous", "same_shape", "rank", "field", "src_absent", "src_present") if key in keys]
        if len(forms) != 1:
            raise ValueError(f"{source}: expected exactly one predicate form")
        form = forms[0]
        if form == "contiguous":
            context.validate_tensor_role(predicate["contiguous"], f"{source}.contiguous")
            if len(keys) != 1:
                raise ValueError(f"{source}: contiguous predicate does not accept extra fields")
        elif form == "same_shape":
            validate_same_shape_predicate(predicate["same_shape"], source, context)
            if len(keys) != 1:
                raise ValueError(f"{source}: same_shape predicate does not accept extra fields")
        elif form == "rank":
            context.validate_tensor_role(predicate["rank"], f"{source}.rank")
            if keys != {"rank", "equals"}:
                raise ValueError(f"{source}: rank predicate requires equals and no other fields")
            if type(predicate["equals"]) is not int or predicate["equals"] < 0:
                raise ValueError(f"{source}.equals: expected non-negative integer")
        elif form == "field":
            validate_field_predicate(predicate, source, context)
        elif form in {"src_absent", "src_present"}:
            context.validate_tensor_role(predicate[form], f"{source}.{form}", allow_optional=True, require_input=True)
            if len(keys) != 1:
                raise ValueError(f"{source}: {form} predicate does not accept extra fields")


def validate_same_shape_predicate(value, source, context):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{source}.same_shape: expected two tensor roles")
    context.validate_tensor_role(value[0], f"{source}.same_shape[0]")
    context.validate_tensor_role(value[1], f"{source}.same_shape[1]")


def validate_field_predicate(predicate, source, context):
    field_type = context.resolve_source(require_string(predicate, "field", source), f"{source}.field")
    comparators = [key for key in ("equals", "in", "min", "max", "multiple_of") if key in predicate]
    if len(comparators) != 1:
        raise ValueError(f"{source}: field predicate requires exactly one comparator")
    comparator = comparators[0]
    value = predicate[comparator]
    if comparator == "equals":
        validate_comparator_value(value, field_type, context, f"{source}.equals")
    elif comparator == "in":
        if not isinstance(value, list) or not value:
            raise ValueError(f"{source}.in: expected non-empty array")
        for i, item in enumerate(value):
            validate_comparator_value(item, field_type, context, f"{source}.in[{i}]")
    elif comparator in {"min", "max"}:
        if field_type not in INTEGER_TYPES and field_type not in {"f32", "f64"}:
            raise ValueError(f"{source}.{comparator}: expected numeric field")
        validate_comparator_value(value, field_type, context, f"{source}.{comparator}")
    elif comparator == "multiple_of":
        if field_type not in INTEGER_TYPES:
            raise ValueError(f"{source}.multiple_of: expected integer field")
        if type(value) is not int or value <= 0:
            raise ValueError(f"{source}.multiple_of: expected positive integer")


def validate_comparator_value(value, field_type, context, source):
    if is_source_string(value):
        value_type = context.resolve_source(value, source)
        if value_type != field_type:
            raise ValueError(f"{source}: source type {value_type} does not match field type {field_type}")
    else:
        validate_literal(value, field_type, source)


def validate_invocation(route, route_path, definition, context):
    invocation = require_dict(route, "invocation", route_path)
    unknown_fields(invocation, INVOCATION_FIELDS, f"{route_path}: invocation")
    validate_buffers(require_array(invocation, "buffers", f"{route_path}: invocation"), route_path, definition, context)
    validate_scalars(require_array(invocation, "scalars", f"{route_path}: invocation"), route_path, definition, context)
    validate_dispatch(require_dict(invocation, "dispatch", f"{route_path}: invocation"), route_path, definition, context)


def validate_buffers(buffers, route_path, definition, context):
    validate_positions(buffers, "invocation.buffers", route_path)
    bindings = require_array(definition, "bindings", route_path)
    binding_by_name = {binding["name"]: binding for binding in bindings}
    seen = set()
    for i, buffer in enumerate(buffers):
        source = f"{route_path}: invocation.buffers[{i}]"
        if not isinstance(buffer, dict):
            raise ValueError(f"{source}: expected object")
        unknown_fields(buffer, BUFFER_FIELDS, source)
        name = require_string(buffer, "name", source)
        binding = binding_by_name.get(name)
        if binding is None:
            raise ValueError(f"{source}: buffer name {name} does not match a definition binding")
        if name in seen:
            raise ValueError(f"{source}: duplicate buffer name {name}")
        seen.add(name)
        tensor = require_string(buffer, "tensor", source)
        context.validate_tensor_role(tensor, f"{source}.tensor")
        kind = require_string(buffer, "kind", source)
        access = BUFFER_KIND_ACCESS.get(kind)
        if access is None:
            raise ValueError(f"{source}: unsupported buffer kind {kind}")
        expected_access = require_string(binding, "access", source)
        if access != expected_access:
            raise ValueError(f"{source}: kind {kind} does not match binding access {expected_access}")
    missing = sorted(set(binding_by_name) - seen)
    if missing:
        raise ValueError(f"{route_path}: invocation.buffers missing bindings {', '.join(missing)}")


def validate_scalars(scalars, route_path, definition, context):
    validate_positions(scalars, "invocation.scalars", route_path)
    parameters = require_array(definition, "parameters", route_path)
    parameter_by_name = {parameter["name"]: parameter for parameter in parameters}
    seen = set()
    for i, scalar in enumerate(scalars):
        source = f"{route_path}: invocation.scalars[{i}]"
        if not isinstance(scalar, dict):
            raise ValueError(f"{source}: expected object")
        unknown_fields(scalar, SCALAR_FIELDS, source)
        name = require_string(scalar, "name", source)
        parameter = parameter_by_name.get(name)
        if parameter is None:
            raise ValueError(f"{source}: scalar name {name} does not match a definition parameter")
        if name in seen:
            raise ValueError(f"{source}: duplicate scalar name {name}")
        seen.add(name)
        scalar_type = require_string(scalar, "type", source)
        parameter_type = require_string(parameter, "type", source)
        if scalar_type != parameter_type:
            raise ValueError(f"{source}: type {scalar_type} does not match parameter type {parameter_type}")
        has_source = "source" in scalar
        has_value = "value" in scalar
        if has_source == has_value:
            raise ValueError(f"{source}: expected exactly one of source or value")
        if has_source:
            value_type = context.resolve_source(require_string(scalar, "source", source), f"{source}.source")
            if value_type != scalar_type:
                raise ValueError(f"{source}.source: source type {value_type} does not match scalar type {scalar_type}")
        else:
            validate_literal(scalar["value"], scalar_type, f"{source}.value")
    missing = sorted(set(parameter_by_name) - seen)
    if missing:
        raise ValueError(f"{route_path}: invocation.scalars missing parameters {', '.join(missing)}")

    abi = require_dict(definition, "abi", route_path)
    constant_byte_length = require_int(abi, "constant_byte_length", route_path)
    packed_size = packed_parameter_size(scalars, route_path)
    if packed_size != constant_byte_length:
        raise ValueError(
            f"{route_path}: invocation scalar packed size {packed_size} does not match "
            f"definition abi.constant_byte_length {constant_byte_length}")


def validate_dispatch(dispatch, route_path, definition, context):
    unknown_fields(dispatch, DISPATCH_FIELDS, f"{route_path}: invocation.dispatch")
    kind = require_string(dispatch, "kind", f"{route_path}: invocation.dispatch")
    workgroup_size = dispatch.get("workgroup_size")
    validate_workgroup_size(workgroup_size, f"{route_path}: invocation.dispatch")
    definition_workgroup_size = require_array(definition, "workgroup_size", route_path)
    if workgroup_size != definition_workgroup_size:
        raise ValueError(f"{route_path}: invocation.dispatch.workgroup_size does not match definition workgroup_size")
    if kind == "flat_1d":
        if set(dispatch) != {"kind", "work_items", "workgroup_size"}:
            raise ValueError(f"{route_path}: flat_1d dispatch requires kind, work_items, and workgroup_size")
        validate_integer_source(dispatch["work_items"], context, f"{route_path}: invocation.dispatch.work_items")
    elif kind == "rows_1d":
        if set(dispatch) != {"kind", "rows", "workgroup_size"}:
            raise ValueError(f"{route_path}: rows_1d dispatch requires kind, rows, and workgroup_size")
        validate_integer_source(dispatch["rows"], context, f"{route_path}: invocation.dispatch.rows")
    elif kind == "exact_3d":
        if set(dispatch) != {"kind", "workgroup_count", "workgroup_size"}:
            raise ValueError(f"{route_path}: exact_3d dispatch requires kind, workgroup_count, and workgroup_size")
        workgroup_count = require_dict(dispatch, "workgroup_count", f"{route_path}: invocation.dispatch")
        unknown_fields(workgroup_count, WORKGROUP_COUNT_FIELDS, f"{route_path}: invocation.dispatch.workgroup_count")
        if set(workgroup_count) != WORKGROUP_COUNT_FIELDS:
            raise ValueError(f"{route_path}: invocation.dispatch.workgroup_count requires x, y, and z")
        for axis in ("x", "y", "z"):
            validate_integer_operand(workgroup_count[axis], context, f"{route_path}: invocation.dispatch.workgroup_count.{axis}")
    else:
        raise ValueError(f"{route_path}: unsupported dispatch kind {kind}")


def validate_integer_source(value, context, source):
    source_type = context.resolve_source(value, source)
    if source_type not in INTEGER_TYPES:
        raise ValueError(f"{source}: expected integer source")


def validate_integer_operand(value, context, source):
    if is_source_string(value):
        validate_integer_source(value, context, source)
    elif type(value) is not int or value <= 0:
        raise ValueError(f"{source}: expected positive integer literal or integer source")
