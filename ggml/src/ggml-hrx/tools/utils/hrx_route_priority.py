from dataclasses import dataclass
from pathlib import Path

from utils import hrx_route_schema as route_schema


FUSION_ROUTE_SCHEMA_V2 = "ggml-hrx-loom-fusion-route-v2"

OPERATION_DIRS = {
    "add",
    "argsort",
    "clamp",
    "copy",
    "div",
    "flash_attn_ext",
    "get_rows",
    "mul",
    "mul_mat",
    "mul_mat_id",
    "rms_norm",
    "rms_norm_mul",
    "rope",
    "set_rows",
    "soft_max",
    "sub",
    "sum_rows",
    "swiglu",
}

FUSION_BUCKETS = (
    (10, 3000, "large fusion"),
    (3, 2000, "medium fusion"),
    (2, 1000, "small fusion"),
)

@dataclass(frozen=True)
class RoutePriority:
    priority: int
    reasons: list[str]


def _route_parts(route_path):
    parts = Path(route_path).parts
    if len(parts) >= 2 and parts[0] == "routes":
        return parts
    raise ValueError(f"{route_path}: expected route path under routes/")


def _is_architecture_root(root):
    return root.startswith("gfx")


def _is_model_path(route_path):
    parts = _route_parts(route_path)
    if len(parts) < 5:
        return False
    root = parts[1]
    return _is_architecture_root(root) and parts[2] not in OPERATION_DIRS


def _is_generic_path(route_path):
    return _route_parts(route_path)[1] == "generic"


def _is_gfx_path(route_path):
    parts = _route_parts(route_path)
    return _is_architecture_root(parts[1])


def _fusion_ops(route, route_path):
    if route_schema.route_schema(route, route_path) != FUSION_ROUTE_SCHEMA_V2:
        return []
    match = route_schema.require_dict(route, "match", route_path)
    ops = route_schema.require_dict(match, "ops", f"{route_path}: match")
    return list(ops)


def _predicate_count(route):
    match = route.get("match")
    if not isinstance(match, dict):
        return 0
    predicates = match.get("predicates", [])
    return len(predicates) if isinstance(predicates, list) else 0


def _dispatch_count(route):
    dispatches = route.get("dispatches", [])
    return len(dispatches) if isinstance(dispatches, list) else 0


def _terminal_output_transient_count(route):
    produced = []
    consumed = set()
    for dispatch in route.get("dispatches", []):
        if not isinstance(dispatch, dict):
            continue
        for buffer in dispatch.get("buffers", []):
            if not isinstance(buffer, dict):
                continue
            transient = buffer.get("transient")
            if not isinstance(transient, str):
                continue
            kind = buffer.get("kind")
            if kind in {"input", "inout"}:
                consumed.add(transient)
            if kind == "output":
                produced.append(transient)
    return sum(1 for transient in produced if transient not in consumed)


def _fusion_bucket(op_count):
    for min_ops, bonus, name in FUSION_BUCKETS:
        if op_count >= min_ops:
            return bonus, name
    if op_count > 0:
        return 0, "single op"
    return 0, "single op"


def _is_gfx_specific_path(route_path, route):
    if not _is_gfx_path(route_path):
        return False
    parts = _route_parts(route_path)
    architecture = parts[1]
    route_id = route_schema.require_string(route, "id", route_path)
    return architecture in Path(route_path).stem or architecture in route_id


def _scope_base(route_path, route, is_fusion, reasons):
    if _is_model_path(route_path) and is_fusion:
        reasons.append("scope: model fusion")
        return 900000
    if _is_model_path(route_path):
        reasons.append("scope: gfx-specific model route")
        return 800000
    if _is_gfx_specific_path(route_path, route):
        reasons.append("scope: gfx-specific route")
        return 800000
    if _is_gfx_path(route_path):
        reasons.append("scope: gfx-generic route")
        return 700000
    if _is_generic_path(route_path):
        reasons.append("scope: generic route")
        return 600000
    reasons.append("scope: generic fallback")
    return 500000


def route_priority(route_path, route):
    reasons = []
    op_count = len(_fusion_ops(route, route_path))
    is_fusion = op_count > 0
    priority = _scope_base(route_path, route, is_fusion, reasons)

    fusion_bonus, fusion_name = _fusion_bucket(op_count)
    priority += fusion_bonus
    if is_fusion:
        reasons.append(f"fusion level: {fusion_name} ({op_count} ops)")
        priority += min(op_count, 99) * 10
    else:
        reasons.append("fusion level: single op")

    match = route.get("match", {})
    if isinstance(match, dict) and match.get("reorder") is True:
        priority += 50
        reasons.append("reorder enabled")

    dispatch_count = _dispatch_count(route)
    if dispatch_count > 1:
        priority += min(dispatch_count, 9) * 5
        reasons.append(f"dispatches: {dispatch_count}")

    predicates = _predicate_count(route)
    if predicates:
        priority += min(predicates, 99)
        reasons.append(f"predicates: {predicates}")

    terminal_output_transients = _terminal_output_transient_count(route)
    if terminal_output_transients:
        priority += min(terminal_output_transients, 9) * 100
        reasons.append(f"terminal output transients: {terminal_output_transients}")

    route_id = route_schema.require_string(route, "id", route_path)
    stable_tiebreak = sum(ord(ch) for ch in route_id) % 10
    priority += stable_tiebreak
    reasons.append(f"stable tiebreak: {stable_tiebreak}")

    return RoutePriority(priority, reasons)
