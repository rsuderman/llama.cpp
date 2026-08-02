# HRX model locks

The HRX Qwen program is specialized for the exact GGUF artifact recorded in
`qwen3-30b-a3b-instruct-2507-q4-k-m.json`. The lock records both the artifact
identity and the tensor-level type/size contract consumed by the owned runtime.

Validate an existing model before using it for graph fixtures:

```sh
python3 ggml/src/ggml-hrx/tools/validate_qwen_model.py \
  --lock ggml/src/ggml-hrx/models/qwen3-30b-a3b-instruct-2507-q4-k-m.json \
  /path/to/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
```

For live graph extraction, use the gated capture wrapper. It performs the full
SHA-256 and tensor-contract validation before launching the planning-only
capture binary:

```sh
python3 ggml/src/ggml-hrx/tools/capture_qwen_graph.py \
  --capture-binary build/hrx-graph-owner/bin/hrx-capture-decode-graph \
  --dump-dir /path/to/capture \
  /path/to/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf prefill-512
```

The current program must not be qualified with similarly named dynamic or
importance-matrix variants such as `UD-Q4_K_XL`; their tensor quantization mix
is materially different.
