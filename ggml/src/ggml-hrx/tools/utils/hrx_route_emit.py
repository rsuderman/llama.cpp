import json
import re


ATTRIBUTE_INDICES = {
    "GGML_OP_ARGSORT": {
        "order": 0,
    },
    "GGML_OP_CLAMP": {
        "minimum": 0,
        "maximum": 1,
    },
    "GGML_OP_FLASH_ATTN_EXT": {
        "scale": 0,
        "max_bias": 1,
        "logit_softcap": 2,
        "precision": 3,
    },
    "GGML_OP_GLU": {
        "glu_op": 0,
    },
    "GGML_OP_ROPE": {
        "n_dims": 1,
        "mode": 2,
        "n_ctx_orig": 4,
        "freq_base": 5,
        "freq_scale": 6,
        "ext_factor": 7,
        "attn_factor": 8,
        "beta_fast": 9,
        "beta_slow": 10,
    },
    "GGML_OP_RMS_NORM": {
        "eps": 0,
    },
    "GGML_OP_SCALE": {
        "scale": 0,
        "bias": 1,
    },
    "GGML_OP_SOFT_MAX": {
        "scale": 0,
        "max_bias": 1,
    },
}

CPP_SCALAR_TYPES = {
    "f32": "float",
    "f64": "double",
    "i32": "int32_t",
    "i64": "int64_t",
    "index": "int32_t",
}

CPP_DTYPE_NAMES = {
    "BF16": "GGML_TYPE_BF16",
    "F16": "GGML_TYPE_F16",
    "F32": "GGML_TYPE_F32",
    "I32": "GGML_TYPE_I32",
    "I64": "GGML_TYPE_I64",
    "Q4_K": "GGML_TYPE_Q4_K",
    "Q5_K": "GGML_TYPE_Q5_K",
    "Q6_K": "GGML_TYPE_Q6_K",
    "Q8_0": "GGML_TYPE_Q8_0",
}

ATTRIBUTE_GETTERS = {
    "f32": "ggml_get_op_params_f32",
    "f64": "ggml_get_op_params_f32",
    "i32": "ggml_get_op_params_i32",
    "i64": "ggml_get_op_params_i32",
    "index": "ggml_get_op_params_i32",
}


def cpp_string(value):
    return json.dumps(value)


def c_identifier(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def role_expr(role):
    if role == "dst":
        return "node"
    if role.startswith("src") and role[3:].isdigit():
        return f"node->src[{int(role[3:])}]"
    raise ValueError(f"unsupported tensor role {role}")


def role_var(role):
    return c_identifier(role)


def derived_var(name):
    return f"derived_{c_identifier(name)}"


def attribute_var(name):
    return f"attribute_{c_identifier(name)}"


def fusion_attribute_var(op_name, name):
    return f"attribute_{c_identifier(op_name)}_{c_identifier(name)}"


def shape_var(role, name):
    return f"shape_{c_identifier(role)}_{c_identifier(name)}"


def scalar_literal(value, scalar_type):
    def float_text(number, precision):
        text = f"{float(number):.{precision}g}"
        if "e" not in text and "E" not in text and "." not in text:
            text += ".0"
        return text

    if scalar_type == "f32":
        return f"{float_text(value, 9)}f"
    if scalar_type == "f64":
        return float_text(value, 17)
    if scalar_type in {"i32", "i64", "index"}:
        return str(int(value))
    raise ValueError(f"unsupported scalar literal type {scalar_type}")


def emit_tensor_type_source(tensor, parts):
    if len(parts) != 3:
        raise ValueError("tensor type source does not accept an index")
    return f"{tensor}->type"


def emit_tensor_rank_source(tensor, parts):
    if len(parts) != 3:
        raise ValueError("tensor rank source does not accept an index")
    return f"ggml_n_dims({tensor})"


def emit_tensor_element_count_source(tensor, parts):
    if len(parts) != 3:
        raise ValueError("tensor element_count source does not accept an index")
    return f"ggml_nelements({tensor})"


def emit_tensor_vector_source(tensor, parts, field):
    if len(parts) == 3:
        field_name = "ne" if field == "dimensions" else "nb"
        return [f"{tensor}->{field_name}[{i}]" for i in range(4)]
    field_name = "ne" if field == "dimensions" else "nb"
    return f"{tensor}->{field_name}[{int(parts[3])}]"


def emit_tensor_dimensions_source(tensor, parts):
    return emit_tensor_vector_source(tensor, parts, "dimensions")


def emit_tensor_strides_source(tensor, parts):
    return emit_tensor_vector_source(tensor, parts, "strides")


def emit_tensor_element_strides_source(tensor, parts):
    if len(parts) == 3:
        return [
            f"({tensor}->nb[{i}] / ggml_type_size({tensor}->type))"
            for i in range(4)
        ]
    return f"({tensor}->nb[{int(parts[3])}] / ggml_type_size({tensor}->type))"


def emit_tensor_permutation_source(tensor, parts):
    if len(parts) == 3:
        return [
            f"ggml_backend_hrx_loom_tensor_permutation_axis({tensor}, {i})"
            for i in range(4)
        ]
    return f"ggml_backend_hrx_loom_tensor_permutation_axis({tensor}, {int(parts[3])})"


def emit_tensor_view_source_source(tensor, parts):
    if len(parts) != 3:
        raise ValueError("tensor view_source source does not accept an index")
    return f"({tensor}->view_src ? {tensor}->view_src : nullptr)"


def emit_tensor_view_offset_bytes_source(tensor, parts):
    if len(parts) != 3:
        raise ValueError("tensor view_offset_bytes source does not accept an index")
    return f"static_cast<int64_t>({tensor}->view_offs)"


TENSOR_FIELD_EMITTERS = {
    "type": emit_tensor_type_source,
    "rank": emit_tensor_rank_source,
    "element_count": emit_tensor_element_count_source,
    "dimensions": emit_tensor_dimensions_source,
    "strides": emit_tensor_strides_source,
    "element_strides": emit_tensor_element_strides_source,
    "permutation": emit_tensor_permutation_source,
    "view_source": emit_tensor_view_source_source,
    "view_offset_bytes": emit_tensor_view_offset_bytes_source,
}


def emit_tensor_source(parts):
    if len(parts) == 2:
        return role_var(parts[1])
    if len(parts) not in {3, 4}:
        raise ValueError("unsupported tensor source")
    role = parts[1]
    field = parts[2]
    emitter = TENSOR_FIELD_EMITTERS.get(field)
    if emitter is None:
        raise ValueError(f"unsupported tensor source field {field}")
    return emitter(role_var(role), parts)


def emit_attribute_source(parts):
    if len(parts) == 2:
        return attribute_var(parts[1])
    if len(parts) == 3:
        return fusion_attribute_var(parts[1], parts[2])
    else:
        raise ValueError("unsupported attribute source")


def emit_derived_source(parts):
    if len(parts) != 2:
        raise ValueError("unsupported derived source")
    return derived_var(parts[1])


def emit_shape_source(parts):
    if len(parts) != 3:
        raise ValueError("unsupported shape source")
    return shape_var(parts[1], parts[2])


SOURCE_EMITTERS = {
    "tensor": emit_tensor_source,
    "attribute": emit_attribute_source,
    "derived": emit_derived_source,
    "shape": emit_shape_source,
}


def source_expr(source):
    parts = source.split(".")
    emitter = SOURCE_EMITTERS.get(parts[0])
    if emitter is None:
        raise ValueError(f"unsupported source {source}")
    return emitter(parts)


def source_type(source, context):
    return context.resolve_source(source, source)


def typed_expr(expr, scalar_type):
    return f"static_cast<{CPP_SCALAR_TYPES[scalar_type]}>({expr})"


def scalar_offsets(scalars, route_path, schema):
    offsets = []
    offset = 0
    for scalar in scalars:
        scalar_type = schema.require_string(scalar, "type", route_path)
        size, alignment = schema.SUPPORTED_SCALAR_TYPES[scalar_type]
        offset = schema.align_offset(offset, alignment)
        offsets.append(offset)
        offset += size
    return offsets, offset


def validate_attribute_indices(route, route_path, schema, attribute_indices):
    match = schema.require_dict(route, "match", route_path)
    if "ops" in match:
        for op_name, op_match in schema.require_dict(match, "ops", route_path).items():
            op = schema.require_string(op_match, "op", f"{route_path}: match.ops.{op_name}")
            attributes = op_match.get("attributes", {})
            indices = attribute_indices.get(op, {})
            for name in attributes:
                if name not in indices:
                    raise ValueError(f"{route_path}: no generated C++ attribute index for {op}.{name}")
        return
    op = schema.require_string(match, "op", f"{route_path}: match")
    attributes = match.get("attributes", {})
    indices = attribute_indices.get(op, {})
    for name in attributes:
        if name not in indices:
            raise ValueError(f"{route_path}: no generated C++ attribute index for {op}.{name}")


def emit_tensor_setup(lines, route, op_rule, schema, route_response, unsupported_shape_reason):
    tensors = schema.route_tensors(route, "route")
    for role in tensors:
        lines.append(f"    const ggml_tensor * {role_var(role)} = {role_expr(role)};")

    required = sorted(op_rule["required_tensors"] & set(tensors))
    if required:
        condition = " || ".join([f"!{role_var(role)}" for role in required])
        lines.extend([
            f"    if ({condition}) {{",
            f"        {route_response(unsupported_shape_reason)}",
            "    }",
        ])

    optional_present = []
    for role, tensor in tensors.items():
        if role in op_rule["optional_tensors"] and tensor.get("optional", False):
            optional_present.append(role)
    for role in optional_present:
        lines.append(f"    const bool {role_var(role)}_present = {role_var(role)} != nullptr;")


def emit_dtype_checks(lines, route, schema, route_response, unsupported_dtype_reason):
    tensors = schema.route_tensors(route, "route")
    checks = []
    for role, tensor in tensors.items():
        dtype = schema.require_string(tensor, "type", "route")
        cpp_dtype = CPP_DTYPE_NAMES[dtype]
        if tensor.get("optional", False):
            checks.append(f"({role_var(role)} && {role_var(role)}->type != {cpp_dtype})")
        else:
            checks.append(f"{role_var(role)}->type != {cpp_dtype}")
    if checks:
        lines.extend([
            f"    if ({' || '.join(checks)}) {{",
            f"        {route_response(unsupported_dtype_reason)}",
            "    }",
        ])


def emit_attributes(lines, route, schema, attribute_indices):
    match = schema.require_dict(route, "match", "route")
    op = schema.require_string(match, "op", "route")
    attributes = match.get("attributes", {})
    if not attributes:
        return
    indices = attribute_indices[op]
    for name, attr in attributes.items():
        scalar_type = schema.require_string(attr, "type", "route")
        getter = ATTRIBUTE_GETTERS[scalar_type]
        cast = CPP_SCALAR_TYPES[scalar_type]
        lines.append(f"    const {cast} {attribute_var(name)} = {getter}(node, {indices[name]});")


def emit_shape_captures(lines, route, schema):
    tensors = schema.route_tensors(route, "route")
    for role, tensor in tensors.items():
        for i, name in enumerate(tensor.get("shape", [])):
            if isinstance(name, str):
                lines.append(f"    const int64_t {shape_var(role, name)} = static_cast<int64_t>({role_var(role)}->ne[{i}]);")


def emit_tensor_declaration_checks(lines, route, schema, route_response, unsupported_shape_reason, unsupported_layout_reason):
    tensors = schema.route_tensors(route, "route")
    enforce_shape_symbols = not schema.route_is_v1(route, "route")
    shape_symbols = {}
    for role, tensor in tensors.items():
        for i, dim in enumerate(tensor.get("shape", [])):
            if isinstance(dim, str):
                if enforce_shape_symbols:
                    value = shape_var(role, dim)
                    if dim in shape_symbols:
                        lines.extend([
                            f"    if ({value} != {shape_symbols[dim]}) {{",
                            f"        {route_response(unsupported_shape_reason)}",
                            "    }",
                        ])
                    else:
                        shape_symbols[dim] = value
            else:
                lines.extend([
                    f"    if ({role_var(role)}->ne[{i}] != {int(dim)}) {{",
                    f"        {route_response(unsupported_shape_reason)}",
                    "    }",
                ])
        layout = tensor.get("layout")
        if layout == "contiguous":
            lines.extend([
                f"    if (!ggml_is_contiguous({role_var(role)})) {{",
                f"        {route_response(unsupported_layout_reason)}",
                "    }",
            ])
        elif layout is not None:
            raise ValueError(f"unsupported tensor layout {layout}")


def emit_integer_operand(value, context, schema):
    if schema.is_source_string(value):
        return source_expr(value)
    return str(int(value))


def emit_integer_operands(value, scalar_type, operator):
    operands = [source_expr(item) if not isinstance(item, int) else str(item) for item in value]
    return f" {operator} ".join([typed_expr(operand, scalar_type) for operand in operands])


def emit_sum(value, scalar_type):
    return emit_integer_operands(value, scalar_type, "+")


def emit_difference(value, scalar_type):
    return emit_integer_operands(value, scalar_type, "-")


def emit_maximum(value, scalar_type):
    operands = [typed_expr(source_expr(item) if not isinstance(item, int) else str(item), scalar_type) for item in value]
    expr = operands[0]
    for operand in operands[1:]:
        expr = f"std::max({expr}, {operand})"
    return expr


def emit_product(value, context, scalar_type):
    if isinstance(value, str):
        operands = source_expr(value)
    else:
        operands = [source_expr(item) if not isinstance(item, int) else str(item) for item in value]
    expr = " * ".join([typed_expr(operand, scalar_type) for operand in operands])
    return expr if expr else scalar_literal(1, scalar_type)


def emit_ceil_div(value, context, scalar_type, schema):
    lhs = emit_integer_operand(value[0], context, schema)
    rhs = emit_integer_operand(value[1], context, schema)
    return typed_expr(f"({lhs} + {rhs} - 1) / {rhs}", scalar_type)


def emit_next_power_of_2(value, context, scalar_type, next_power_of_2_function):
    del context
    expr = source_expr(value)
    return typed_expr(f"{next_power_of_2_function}({expr})", scalar_type)


def emit_derived_field(item, context, scalar_type, schema, next_power_of_2_function):
    del context
    del next_power_of_2_function
    expr = source_expr(schema.require_string(item, "field", "route"))
    return typed_expr(expr, scalar_type)


def emit_derived_value(item, context, scalar_type, schema, next_power_of_2_function):
    del context
    del schema
    del next_power_of_2_function
    return scalar_literal(item["value"], scalar_type)


def emit_derived_sum(item, context, scalar_type, schema, next_power_of_2_function):
    del context
    del schema
    del next_power_of_2_function
    return emit_sum(item["sum"], scalar_type)


def emit_derived_difference(item, context, scalar_type, schema, next_power_of_2_function):
    del context
    del schema
    del next_power_of_2_function
    return emit_difference(item["difference"], scalar_type)


def emit_derived_maximum(item, context, scalar_type, schema, next_power_of_2_function):
    del context
    del schema
    del next_power_of_2_function
    return emit_maximum(item["maximum"], scalar_type)


def emit_derived_product(item, context, scalar_type, schema, next_power_of_2_function):
    del schema
    del next_power_of_2_function
    return emit_product(item["product"], context, scalar_type)


def emit_derived_ceil_div(item, context, scalar_type, schema, next_power_of_2_function):
    del next_power_of_2_function
    return emit_ceil_div(item["ceil_div"], context, scalar_type, schema)


def emit_derived_next_power_of_2(item, context, scalar_type, schema, next_power_of_2_function):
    return emit_next_power_of_2(schema.require_string(item, "next_power_of_2", "route"), context, scalar_type, next_power_of_2_function)


DERIVED_EMITTERS = {
    "field": emit_derived_field,
    "value": emit_derived_value,
    "sum": emit_derived_sum,
    "difference": emit_derived_difference,
    "product": emit_derived_product,
    "ceil_div": emit_derived_ceil_div,
    "maximum": emit_derived_maximum,
    "next_power_of_2": emit_derived_next_power_of_2,
}


def emit_derived(lines, route, context, schema, next_power_of_2_function):
    derived = schema.require_dict(route, "derived", "route")
    for name, item in derived.items():
        scalar_type = schema.require_string(item, "type", "route")
        cpp_type = CPP_SCALAR_TYPES[scalar_type]
        operation = next((key for key in DERIVED_EMITTERS if key in item), None)
        if operation is None:
            raise ValueError(f"unsupported derived operation for {name}")
        expr = DERIVED_EMITTERS[operation](item, context, scalar_type, schema, next_power_of_2_function)
        lines.append(f"    const {cpp_type} {derived_var(name)} = {expr};")


def comparator_value_expr(value, field_type, context, schema):
    del context
    if schema.is_source_string(value):
        return source_expr(value)
    if field_type == "dtype":
        return CPP_DTYPE_NAMES[value]
    if field_type in CPP_SCALAR_TYPES:
        return scalar_literal(value, field_type)
    raise ValueError(f"unsupported comparator value type {field_type}")


def emit_equals_comparator(lhs, value, field_type, context, schema):
    return f"{lhs} == {comparator_value_expr(value, field_type, context, schema)}"


def emit_in_comparator(lhs, value, field_type, context, schema):
    return " || ".join([f"{lhs} == {comparator_value_expr(item, field_type, context, schema)}" for item in value])


def emit_min_comparator(lhs, value, field_type, context, schema):
    return f"{lhs} >= {comparator_value_expr(value, field_type, context, schema)}"


def emit_max_comparator(lhs, value, field_type, context, schema):
    return f"{lhs} <= {comparator_value_expr(value, field_type, context, schema)}"


def emit_multiple_of_comparator(lhs, value, field_type, context, schema):
    del field_type
    del context
    del schema
    return f"{lhs} % {int(value)} == 0"


def emit_divisible_by_comparator(lhs, value, field_type, context, schema):
    rhs = comparator_value_expr(value, field_type, context, schema)
    return f"{rhs} != 0 && {lhs} % {rhs} == 0"


COMPARATOR_EMITTERS = {
    "equals": emit_equals_comparator,
    "in": emit_in_comparator,
    "min": emit_min_comparator,
    "max": emit_max_comparator,
    "multiple_of": emit_multiple_of_comparator,
    "divisible_by": emit_divisible_by_comparator,
}


def comparator_expr(lhs, comparator, value, field_type, context, schema):
    emitter = COMPARATOR_EMITTERS.get(comparator)
    if emitter is None:
        raise ValueError(f"unsupported comparator {comparator}")
    return emitter(lhs, value, field_type, context, schema)


def emit_contiguous_predicate(predicate, context, schema):
    del context
    del schema
    return f"ggml_is_contiguous({role_var(predicate['contiguous'])})"


def emit_same_shape_predicate(predicate, context, schema):
    del context
    del schema
    values = predicate["same_shape"]
    if len(values) <= 1:
        return "true"
    lhs = values[0]
    return " && ".join([f"ggml_are_same_shape({role_var(lhs)}, {role_var(rhs)})" for rhs in values[1:]])


def emit_same_layout_predicate(predicate, context, schema):
    del context
    del schema
    values = predicate["same_layout"]
    if len(values) <= 1:
        return "true"
    lhs = values[0]
    return " && ".join([
        f"ggml_backend_hrx_loom_tensors_have_same_layout({role_var(lhs)}, {role_var(rhs)})"
        for rhs in values[1:]
    ])


def emit_rank_predicate(predicate, context, schema):
    del context
    del schema
    return f"ggml_n_dims({role_var(predicate['rank'])}) == {int(predicate['equals'])}"


def emit_field_predicate(predicate, context, schema):
    field = schema.require_string(predicate, "field", "route")
    field_type = source_type(field, context)
    lhs = source_expr(field)
    comparator = next(key for key in COMPARATOR_EMITTERS if key in predicate)
    return comparator_expr(lhs, comparator, predicate[comparator], field_type, context, schema)


def emit_src_absent_predicate(predicate, context, schema):
    del context
    del schema
    return f"{role_expr(predicate['src_absent'])} == nullptr"


def emit_src_present_predicate(predicate, context, schema):
    del context
    del schema
    return f"{role_expr(predicate['src_present'])} != nullptr"


def emit_elided_predicate(predicate, context, schema):
    del context
    del schema
    return " && ".join([
        f"ggml_backend_hrx_loom_tensor_is_transient(request, {role_var(tensor)}, matched_node_indices, matched_node_count)"
        for tensor in predicate["elided"]
    ])


def emit_no_overlap_predicate(predicate, context, schema):
    del context
    del schema
    lhs, rhs = predicate["no_overlap"]
    return f"ggml_backend_hrx_loom_reorder_match(request, plan) || !ggml_backend_hrx_loom_tensors_overlap({role_var(lhs)}, {role_var(rhs)})"


PREDICATE_EMITTERS = {
    "contiguous": emit_contiguous_predicate,
    "same_shape": emit_same_shape_predicate,
    "same_layout": emit_same_layout_predicate,
    "rank": emit_rank_predicate,
    "field": emit_field_predicate,
    "src_absent": emit_src_absent_predicate,
    "src_present": emit_src_present_predicate,
    "elided": emit_elided_predicate,
    "no_overlap": emit_no_overlap_predicate,
}


def emit_predicates(lines, route, context, schema, unsupported_reason_by_predicate, route_response):
    match = schema.require_dict(route, "match", "route")
    for predicate in match.get("predicates", []):
        form = next(key for key in PREDICATE_EMITTERS if key in predicate)
        reason = unsupported_reason_by_predicate[form]
        condition = PREDICATE_EMITTERS[form](predicate, context, schema)
        lines.extend([
            f"    if (!({condition})) {{",
            f"        {route_response(reason)}",
            "    }",
        ])


def dispatch_axis_exprs(values, context, schema):
    axes = [emit_integer_operand(value, context, schema) for value in values]
    while len(axes) < 3:
        axes.append("1")
    return axes


def emit_dispatch_config(dispatch, context, route_constant, failed_response, schema):
    if "work_items" in dispatch:
        helper = "ggml_backend_hrx_make_work_items_dispatch_config"
        field = "work_items"
        values = schema.require_array(dispatch, field, "route")
        value_name = "dispatch_work_items"
    else:
        helper = "ggml_backend_hrx_make_workgroup_dispatch_config"
        field = "workgroups"
        values = schema.require_array(dispatch, field, "route")
        value_name = "dispatch_workgroups"
    x, y, z = dispatch_axis_exprs(values, context, schema)
    return [
        f"    const int64_t {value_name}[3] = {{{x}, {y}, {z}}};",
        f"    if (!{helper}(entry->workgroup_size, {value_name}, &plan->dispatch)) {{",
        f"        {failed_response(route_constant)}",
        "    }",
    ]


def anchor_op(route, schema):
    match = schema.require_dict(route, "match", "route")
    if "op" in match:
        return schema.require_string(match, "op", "route")
    ops = schema.require_dict(match, "ops", "route")
    anchors = match.get("anchors", [])
    if anchors:
        return schema.require_string(ops[anchors[0]], "op", "route")
    first_op = next(iter(ops.values()))
    return schema.require_string(first_op, "op", "route")


def route_infos_for_dispatcher(routes, schema):
    route_infos = []
    for route_path, route, _ in routes:
        op = anchor_op(route, schema)
        route_id = schema.require_string(route, "id", "route")
        priority = schema.require_int(route, "priority", route_path)
        route_infos.append((op, -priority, route_id, route))
    route_infos.sort()
    return route_infos


def ops_from_route_infos(route_infos):
    ops = []
    for op, _, _, _ in route_infos:
        if op not in ops:
            ops.append(op)
    return ops


def route_infos_for_op(routes, selected_op, schema):
    route_infos = []
    for route_path, route, definition in routes:
        op = anchor_op(route, schema)
        if op != selected_op:
            continue
        route_id = schema.require_string(route, "id", "route")
        priority = schema.require_int(route, "priority", route_path)
        route_infos.append((-priority, route_id, route_path, route, definition))
    route_infos.sort()
    return route_infos
