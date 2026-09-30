# HRX backend compatibility

`hrx-minimum-commit.txt` records the minimum required commit in ROCm/hrx-system. It contains one full Git commit SHA. The backend's C++ APIs, Loom syntax, and numerical behavior can all depend on that revision.

When `GGML_HRX=ON` and `HRX_SOURCE_DIR` is set, CMake requires the HRX checkout's HEAD to equal or descend from the recorded commit. The check uses local Git ancestry, not dates or version strings. It supports submodules and worktrees. Local edits are allowed but are not verified by the ancestry check.

A missing commit, insufficient shallow history, or a source archive cannot establish compatibility and causes configuration to fail. Supply a Git checkout with sufficient history and rerun CMake. Configuration never fetches history or changes the checkout. Installed HRX/Loom packages remain outside this check.

## Updating the minimum

Raise the minimum in the same llama.cpp commit that introduces a newer HRX dependency. This includes new API calls, new Loom syntax, and removing a numerical workaround after an HRX compiler or runtime fix. Unrelated repairs do not require a bump.

Choose the earliest demonstrated required commit that retains the previous floor in its ancestry. Do not replace the floor with the latest tested revision merely because it is available. Record the dependency and its HRX commit or PR in the change's validation evidence. A minimum is not a promise that every future HRX revision is compatible.

The initial minimum, `40b7cc13717d9d4c7f7eb0e6885563042153566b` ([HRX #1054](https://github.com/ROCm/hrx-system/pull/1054)), preserves CFG guard proofs against constant bounds. The Q6 prefill JIT case with 512 tokens, 5120 inputs, and 1984 outputs needs that proof to compile its guarded fragment stores. The preceding commit fails with `SUBRANGE/010`; this revision passes the JIT compile suite.

This minimum also contains `cba4f16b2902c218be16f5394fad94563eb68173`, which corrects packed AMDGPU table lookup ordering. The IQ4 decoders now use logical table indices without the old XOR-12 compensation and require that correction.

For builds using installed packages, check the source revision used to produce both HRX and Loom against this floor before validating a repair. From the staging checkout:

```sh
git -C hrx-system merge-base --is-ancestor "$(cat llama.cpp/ggml/src/ggml-hrx/hrx-minimum-commit.txt)" HEAD
```

This source check is useful only when the tested libraries were built from that checkout; record the actual artifact provenance as well.

## Configure-check tests

Run `python3 tests/test-hrx-minimum-commit.py` from the llama.cpp root. The tests use temporary Git repositories and CMake projects and require no HRX build or GPU.
