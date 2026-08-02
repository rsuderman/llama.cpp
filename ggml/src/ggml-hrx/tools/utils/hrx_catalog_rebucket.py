import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from utils import hrx_route_schema as route_schema


AMDGPU_TARGET_RE = re.compile(r"amdgpu\.target<([^>]+)>")
EXACT_ARCH_RE = re.compile(r"^gfx[0-9]+$")
GFX11_GENERIC_BUCKET = "gfx11-generic"
GFX11_GENERIC_ARCHITECTURES = ["gfx1100", "gfx1151"]
GFX11_EXACT_ARCHITECTURES = set(GFX11_GENERIC_ARCHITECTURES)
GENERIC_ARCHITECTURES = ["*"]


@dataclass(frozen=True)
class Move:
    old_path: Path
    new_path: Path
    reason: str


@dataclass(frozen=True)
class Replacement:
    path: Path
    old: object
    new: object
    reason: str


@dataclass
class Plan:
    moves: list[Move]
    replacements: list[Replacement]
    issues: list[str]
    notes: list[str]


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def source_targets(source_path):
    text = source_path.read_text(encoding="utf-8")
    return set(AMDGPU_TARGET_RE.findall(text))


def path_root(path):
    parts = Path(path).parts
    if len(parts) < 2:
        return None
    return parts[1]


def is_qwen_moe_source(path):
    parts = Path(path).parts
    if len(parts) < 3 or parts[0] != "sources":
        return False
    return parts[2] == "qwen_moe" or Path(path).name.startswith("qwen3_moe_")


def is_exact_arch(value):
    return bool(EXACT_ARCH_RE.match(value))


def is_hardware_bucket(value):
    return value == GFX11_GENERIC_BUCKET or is_exact_arch(value)


def bucket_specificity(bucket):
    if bucket == "generic":
        return 0
    if bucket == GFX11_GENERIC_BUCKET:
        return 1
    if is_exact_arch(bucket):
        return 2
    return -1


def most_specific_bucket(buckets):
    return max(buckets, key=bucket_specificity)


def target_bucket(targets, fallback_bucket):
    if not targets:
        return fallback_bucket
    if len(targets) != 1:
        return None
    return next(iter(targets))


def collapse_duplicate_dirs(parts):
    collapsed = []
    for part in parts:
        if collapsed and collapsed[-1] == part:
            continue
        collapsed.append(part)
    return collapsed


def canonical_bucket_path(path, bucket):
    parts = list(Path(path).parts)
    if len(parts) < 3:
        return Path(path)
    root = parts[0]
    if root not in {"sources", "defs", "routes"}:
        return Path(path)
    rest = collapse_duplicate_dirs(parts[2:])
    return Path(root, bucket, *rest)


def relative_to_catalog(path):
    return path.as_posix()


def route_names(source_root):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    return route_schema.require_list(metadata, "routes", metadata_path)


def metadata_issues(source_root):
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    issues = []
    for target in route_schema.require_list(metadata, "targets", metadata_path):
        if target == GFX11_GENERIC_BUCKET:
            issues.append(f"{metadata_path.relative_to(source_root)}: metadata targets must be exact hardware, not {GFX11_GENERIC_BUCKET}")
        elif target != "generic" and not is_exact_arch(target):
            issues.append(f"{metadata_path.relative_to(source_root)}: unsupported metadata target {target}")
    return issues


def source_inventory(source_root):
    inventory = {}
    issues = []
    for source_path in sorted((source_root / "sources").rglob("*.loom")):
        rel = source_path.relative_to(source_root)
        root = path_root(rel)
        targets = source_targets(source_path)
        if len(targets) > 1:
            issues.append(f"{rel}: mixed AMDGPU targets {', '.join(sorted(targets))}")
        if is_qwen_moe_source(rel):
            bucket = GFX11_GENERIC_BUCKET
            targetless = False
        else:
            bucket = target_bucket(targets, root)
            targetless = not targets
        if bucket and bucket != "generic" and not is_hardware_bucket(bucket):
            issues.append(f"{rel}: unsupported AMDGPU target bucket {bucket}")
        inventory[rel.as_posix()] = {
            "path": rel,
            "root": root,
            "targets": targets,
            "bucket": bucket,
            "targetless": targetless,
        }
    return inventory, issues


def definition_inventory(source_root, sources):
    inventory = {}
    issues = []
    for definition_path in sorted((source_root / "defs").rglob("*.json")):
        rel = definition_path.relative_to(source_root)
        definition = read_json(definition_path)
        source = route_schema.require_string(definition, "source", definition_path)
        source_info = sources.get(source)
        if source_info is None:
            issues.append(f"{rel}: missing source {source}")
            continue
        root = path_root(rel)
        bucket = source_info["bucket"] if not source_info["targetless"] else root
        dependencies = []
        for i, dependency in enumerate(definition.get("dependencies", [])):
            dependency_source = route_schema.require_string(dependency, "source", f"{definition_path}: dependencies[{i}]")
            dependency_info = sources.get(dependency_source)
            if dependency_info is None:
                issues.append(f"{rel}: missing dependency source {dependency_source}")
                continue
            dependencies.append(dependency_info)
            if dependency_info["targetless"]:
                continue
            if bucket_specificity(dependency_info["bucket"]) > bucket_specificity(bucket):
                issues.append(f"{rel}: dependency {dependency_source} is more specific than def bucket {bucket}")
        inventory[rel.as_posix()] = {
            "path": rel,
            "root": root,
            "definition": definition,
            "source": source_info,
            "dependencies": dependencies,
            "bucket": bucket,
        }
    return inventory, issues


def route_inventory(source_root, definitions):
    inventory = {}
    issues = []
    for route_name in route_names(source_root):
        route_path = source_root / route_name
        route = read_json(route_path)
        rel = Path(route_name)
        root = path_root(rel)
        if root not in {"generic", GFX11_GENERIC_BUCKET} and not is_exact_arch(root):
            issues.append(f"{rel}: route bucket must be generic, gfx11-generic, or exact hardware")
        architectures = route_schema.require_list(route, "architectures", route_path)
        for architecture in architectures:
            if architecture == GFX11_GENERIC_BUCKET:
                issues.append(f"{rel}: route architectures must use exact hardware targets, not {GFX11_GENERIC_BUCKET}")
        route_definitions = []
        for i, dispatch in enumerate(route_schema.require_list(route, "dispatches", route_path)):
            definition = route_schema.require_string(dispatch, "definition", f"{route_path}: dispatches[{i}]")
            definition_path = (route_path.parent / definition).resolve().relative_to(source_root.resolve())
            definition_info = definitions.get(definition_path.as_posix())
            if definition_info is None:
                issues.append(f"{rel}: missing dispatch definition {definition_path}")
                continue
            route_definitions.append(definition_info)
        inventory[rel.as_posix()] = {
            "path": rel,
            "root": root,
            "route": route,
            "definitions": route_definitions,
        }
    return inventory, issues


def planned_source_moves(sources):
    moves = []
    for info in sources.values():
        bucket = info["bucket"]
        path = info["path"]
        if info["targetless"] or bucket is None:
            continue
        canonical = canonical_bucket_path(path, bucket)
        if canonical != path:
            moves.append(Move(path, canonical, f"source target bucket is {bucket}"))
    return moves


def planned_definition_moves(definitions, source_moves_by_old):
    moves = []
    for info in definitions.values():
        bucket = info["bucket"]
        path = info["path"]
        if bucket is None:
            continue
        canonical = canonical_bucket_path(path, bucket)
        if canonical != path:
            moves.append(Move(path, canonical, f"definition source bucket is {bucket}"))
    return moves


def route_bucket(route):
    buckets = [definition["bucket"] for definition in route["definitions"] if definition["bucket"]]
    if not buckets:
        return None
    return most_specific_bucket(buckets)


def planned_route_moves(routes):
    moves = []
    for route in routes.values():
        path = route["path"]
        bucket = route_bucket(route)
        if bucket is None or bucket == path_root(path):
            continue
        canonical = canonical_bucket_path(path, bucket)
        if canonical != path:
            moves.append(Move(path, canonical, f"route dispatch definitions resolve to {bucket}"))
    return moves


def replacement_map(moves):
    return {move.old_path.as_posix(): move.new_path.as_posix() for move in moves}


def planned_replacements(source_root, source_moves, definition_moves, route_moves):
    replacements = []
    source_paths = replacement_map(source_moves)
    definition_paths = replacement_map(definition_moves)
    route_paths = replacement_map(route_moves)
    has_gfx11_generic_route = bool(route_paths)
    for source_path in sorted((source_root / "sources").rglob("*.loom")):
        rel = source_path.relative_to(source_root)
        if not is_qwen_moe_source(rel):
            continue
        for target in source_targets(source_path):
            if target in GFX11_EXACT_ARCHITECTURES:
                replacements.append(Replacement(rel, f"amdgpu.target<{target}>", f"amdgpu.target<{GFX11_GENERIC_BUCKET}>", f"qwen_moe sources use {GFX11_GENERIC_BUCKET} targets"))
    for definition_path in sorted((source_root / "defs").rglob("*.json")):
        rel = definition_path.relative_to(source_root)
        definition = read_json(definition_path)
        source = definition.get("source")
        if source in source_paths:
            replacements.append(Replacement(rel, source, source_paths[source], "definition source moved"))
        for dependency in definition.get("dependencies", []):
            dependency_source = dependency.get("source")
            if dependency_source in source_paths:
                replacements.append(Replacement(rel, dependency_source, source_paths[dependency_source], "definition dependency moved"))
    for route_name in route_names(source_root):
        route_path = source_root / route_name
        route = read_json(route_path)
        rel = Path(route_name)
        future_route_name = route_paths.get(route_name, route_name)
        future_route_path = source_root / future_route_name
        if path_root(future_route_name) == GFX11_GENERIC_BUCKET:
            has_gfx11_generic_route = True
        for dispatch in route.get("dispatches", []):
            definition = dispatch.get("definition")
            if not isinstance(definition, str):
                continue
            absolute = (route_path.parent / definition).resolve().relative_to(source_root.resolve()).as_posix()
            future_definition = definition_paths.get(absolute, absolute)
            if future_definition == absolute and future_route_name == route_name:
                continue
            new_abs = (source_root / future_definition).resolve()
            new_rel = os.path.relpath(new_abs, future_route_path.parent.resolve())
            replacements.append(Replacement(rel, definition, Path(new_rel).as_posix(), "route definition moved"))
        if path_root(future_route_name) == GFX11_GENERIC_BUCKET:
            architectures = route.get("architectures")
            if architectures != GFX11_GENERIC_ARCHITECTURES:
                replacements.append(Replacement(rel, architectures, GFX11_GENERIC_ARCHITECTURES, f"route supports {GFX11_GENERIC_BUCKET} architectures"))
        elif path_root(future_route_name) == "generic":
            architectures = route.get("architectures")
            if architectures != GENERIC_ARCHITECTURES:
                replacements.append(Replacement(rel, architectures, GENERIC_ARCHITECTURES, "generic route supports all architectures"))
    metadata_path = source_root / "metadata.json"
    metadata = read_json(metadata_path)
    if has_gfx11_generic_route:
        targets = metadata.get("targets", [])
        new_targets = list(targets)
        for architecture in GFX11_GENERIC_ARCHITECTURES:
            if architecture not in new_targets:
                new_targets.append(architecture)
        if new_targets != targets:
            replacements.append(Replacement(Path("metadata.json"), targets, new_targets, f"metadata lists exact {GFX11_GENERIC_BUCKET} hardware targets"))
    for route_name in metadata.get("routes", []):
        if route_name in route_paths:
            replacements.append(Replacement(Path("metadata.json"), route_name, route_paths[route_name], "metadata route moved"))
    return replacements


def collect_plan(source_root):
    metadata_target_issues = metadata_issues(source_root)
    sources, source_issues = source_inventory(source_root)
    definitions, definition_issues = definition_inventory(source_root, sources)
    routes, route_issues = route_inventory(source_root, definitions)
    source_moves = planned_source_moves(sources)
    definition_moves = planned_definition_moves(definitions, replacement_map(source_moves))
    route_moves = planned_route_moves(routes)
    replacements = planned_replacements(source_root, source_moves, definition_moves, route_moves)
    notes = []
    for route in routes.values():
        if route["root"] == "generic":
            continue
        if not route["definitions"]:
            continue
        buckets = {definition["bucket"] for definition in route["definitions"] if definition["bucket"]}
        if buckets:
            notes.append(f"{route['path']}: route references {most_specific_bucket(buckets)} definitions")
    return Plan(source_moves + definition_moves + route_moves, replacements, metadata_target_issues + source_issues + definition_issues + route_issues, notes)


def apply_replacements(source_root, replacements):
    by_path = {}
    for replacement in replacements:
        by_path.setdefault(replacement.path, []).append(replacement)
    for rel, path_replacements in by_path.items():
        path = source_root / rel
        if path.suffix == ".loom":
            text = path.read_text(encoding="utf-8")
            for replacement in path_replacements:
                if not isinstance(replacement.old, str) or not isinstance(replacement.new, str):
                    raise TypeError(f"{replacement.path}: loom replacements must be strings")
                text = text.replace(replacement.old, replacement.new)
            path.write_text(text, encoding="utf-8")
            continue
        if any(not isinstance(replacement.old, str) or not isinstance(replacement.new, str) for replacement in path_replacements):
            data = read_json(path)
            for replacement in path_replacements:
                data = replace_json_value(data, replacement.old, replacement.new)
            write_json(path, data)
            continue
        text = path.read_text(encoding="utf-8")
        for replacement in path_replacements:
            text = text.replace(json.dumps(replacement.old), json.dumps(replacement.new))
        path.write_text(text, encoding="utf-8")


def replace_json_value(value, old, new):
    if value == old:
        return new
    if isinstance(value, list):
        return [replace_json_value(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_json_value(item, old, new) for key, item in value.items()}
    return value


def apply_moves(source_root, moves):
    for move in moves:
        old_path = source_root / move.old_path
        new_path = source_root / move.new_path
        if old_path == new_path:
            continue
        if new_path.exists():
            raise ValueError(f"{move.new_path}: destination already exists")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))


def remove_empty_dirs(root):
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def apply_plan(source_root, plan, clean_empty_dirs=False):
    if plan.issues:
        raise ValueError("cannot apply rebucket plan with blocking issues")
    apply_replacements(source_root, plan.replacements)
    apply_moves(source_root, plan.moves)
    if clean_empty_dirs:
        remove_empty_dirs(source_root / "sources")
        remove_empty_dirs(source_root / "defs")
