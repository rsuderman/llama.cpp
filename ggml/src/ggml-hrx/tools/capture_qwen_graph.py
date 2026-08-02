#!/usr/bin/env python3
"""Validates the pinned Qwen GGUF before running the planning-only capture."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


TOOLS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LOCK = TOOLS_DIR.parent / "models" / "qwen3-30b-a3b-instruct-2507-q4-k-m.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("workload", choices=("prefill-512", "decode-513"))
    parser.add_argument("--capture-binary", type=pathlib.Path, required=True)
    parser.add_argument("--dump-dir", type=pathlib.Path, required=True)
    parser.add_argument("--lock", type=pathlib.Path, default=DEFAULT_LOCK)
    parser.add_argument("--validation-report", type=pathlib.Path)
    parser.add_argument("--dump-level", default="summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = TOOLS_DIR / "validate_qwen_model.py"
    validation_command = [
        sys.executable,
        str(validator),
        str(args.model),
        "--lock",
        str(args.lock),
    ]
    if args.validation_report:
        validation_command += ["--report", str(args.validation_report)]
    subprocess.run(validation_command, check=True)

    environment = os.environ.copy()
    environment.update({
        "GGML_HRX_DUMP_GRAPH_DIR": str(args.dump_dir.resolve()),
        "GGML_HRX_DUMP_LEVEL": args.dump_level,
        "GGML_HRX_GRAPH_ORACLE": "1",
    })
    completed = subprocess.run(
        [str(args.capture_binary.resolve()), str(args.model.resolve()), args.workload],
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
