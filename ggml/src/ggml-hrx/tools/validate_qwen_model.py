#!/usr/bin/env python3
"""Validates a GGUF artifact against a pinned HRX model and tensor contract."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--lock", type=pathlib.Path, required=True)
    parser.add_argument("--metadata-only", action="store_true",
                        help="skip the full-file SHA-256 check")
    parser.add_argument("--report", type=pathlib.Path)
    return parser.parse_args()


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schema") != "ggml-hrx-model-lock-v1":
        raise RuntimeError("unsupported model lock schema")

    source_root = pathlib.Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(source_root / "gguf-py"))
    from gguf import GGUFReader  # pylint: disable=import-outside-toplevel

    errors: list[str] = []
    artifact = lock["artifact"]
    actual_size = args.model.stat().st_size
    if args.model.name != artifact["filename"]:
        errors.append(f"filename {args.model.name!r} != locked {artifact['filename']!r}")
    if actual_size != artifact["size"]:
        errors.append(f"file size {actual_size} != locked {artifact['size']}")

    actual_sha256 = None
    if not args.metadata_only and actual_size == artifact["size"]:
        actual_sha256 = file_sha256(args.model)
        if actual_sha256 != artifact["sha256"]:
            errors.append(f"SHA-256 {actual_sha256} != locked {artifact['sha256']}")

    reader = GGUFReader(args.model, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected_names: set[str] = set()
    contracts: list[tuple[str, dict[str, int]]] = []
    for name, specification in lock["top_level"].items():
        contracts.append((name, specification["types"]))
    for layer in range(lock["owned_runtime"]["layer_count"]):
        for suffix, specification in lock["per_layer"].items():
            contracts.append((f"blk.{layer}.{suffix}", specification["types"]))

    for name, types in contracts:
        expected_names.add(name)
        tensor = tensors.get(name)
        if tensor is None:
            errors.append(f"missing tensor {name}")
            continue
        tensor_type = tensor.tensor_type.name
        if tensor_type not in types:
            errors.append(f"tensor {name} has {tensor_type}; allowed {','.join(types)}")
        elif tensor.n_bytes != types[tensor_type]:
            errors.append(f"tensor {name} has {tensor.n_bytes} bytes; expected {types[tensor_type]}")

    unexpected = sorted(set(tensors) - expected_names)
    if unexpected:
        errors.append(f"unexpected tensors ({len(unexpected)}): {','.join(unexpected)}")
    expected_count = lock["owned_runtime"]["parameter_count"]
    if len(reader.tensors) != expected_count:
        errors.append(f"tensor count {len(reader.tensors)} != locked {expected_count}")

    type_histogram = collections.Counter(tensor.tensor_type.name for tensor in reader.tensors)
    descriptor_text = "\n".join(
        f"{tensor.name}\t{tensor.tensor_type.name}\t{tensor.n_bytes}\t"
        + ",".join(str(int(value)) for value in tensor.shape)
        for tensor in sorted(reader.tensors, key=lambda item: item.name)
    )
    report = {
        "schema": "ggml-hrx-model-validation-v1",
        "valid": not errors,
        "model": str(args.model.resolve()),
        "lock": str(args.lock.resolve()),
        "artifact_revision": artifact["revision"],
        "size": actual_size,
        "sha256": actual_sha256,
        "tensor_count": len(reader.tensors),
        "tensor_descriptor_sha256": hashlib.sha256(descriptor_text.encode()).hexdigest(),
        "type_histogram": dict(sorted(type_histogram.items())),
        "errors": errors,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
