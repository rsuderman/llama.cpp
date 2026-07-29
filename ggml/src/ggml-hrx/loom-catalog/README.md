# HRX Loom Catalog

This catalog is the initial Loom route catalog for HRX. It mirrors one current HSACO `GGML_OP_ADD` route and adds the required Loom compile `config` section.

## Version

- Catalog metadata schema: `ggml-hrx-loom-catalog-v0`
- Definition schema: `ggml-hrx-loom-def-v1`
- Route schema: `ggml-hrx-loom-route-v1`
- Initial target set: `gfx1100`
- Initial route: `GGML_OP_ADD` F32 contiguous tensors with matching shapes

## Files

- `metadata.json` lists the catalog version, target set, and route files.
- `defs/add_f32.json` describes the Loom source identity and runtime ABI for `hrx_add_f32`.
- `routes/add_f32.json` describes route matching, derived values, compile config, tensor and scalar invocation, and dispatch geometry.
- `sources/add_f32.loom` is the hand-authored Loom source for F32 contiguous ADD.

## Route Shape

The ADD route matches F32 `src0`, `src1`, and `dst` tensors. All three tensors must be contiguous, and `src0`, `src1`, and `dst` must have the same shape.

The route derives `nelements` from the captured destination shape. That value is used as both the runtime scalar parameter and a compile config binding. The route also binds `workgroup_size_x` as a compile config value of 256.

This catalog uses only the final Loom route fields: `match`, `derived`, `config`, and `invocation`. It does not include HSACO compatibility fields.

The `config` section is required. Its `mode` must be `compile`, and `bindings` is an ordered array of compile-time values. Each binding has a C identifier `name`, a scalar `type`, and exactly one of `source` or `value`. The current generated runtime stores at most 16 config bindings, with names limited to 63 bytes and formatted values limited to 127 bytes.

## Validation

Validate the checked-in catalog with:

```sh
python ggml/src/ggml-hrx/tools/validate_loom_routes.py --source-root ggml/src/ggml-hrx/loom-catalog
```

The durable validator smoke test is:

```sh
python ggml/src/ggml-hrx/tools/test_validate_loom_routes.py
```

## Build And Runtime

Loom catalog generation and runtime code are compiled only when HRX is configured with:

```sh
cmake -S . -B build-hrx -DGGML_HRX=ON -DGGML_HRX_LOOM=ON \
  -Dhrx_DIR=/path/to/hrx/libhrx/cmake/hrx \
  -Dloomc_DIR=/path/to/hrx/loom/binding/c/cmake/loomc
```

`GGML_HRX_LOOM=ON` requires both the HRX runtime package and the Loom C API package. The first supported source format is `loom-text`, which is compiled in-process through `loomc` and emitted as AMDGPU HSACO before loading through HRX.

Even in a Loom-enabled build, Loom routing is disabled at runtime unless `GGML_HRX_ENABLE_LOOM` is set. HRX tries HSACO first by default. Set `GGML_HRX_LOOM_FIRST=1` together with `GGML_HRX_ENABLE_LOOM=1` to try Loom routes before HSACO routes during development.

The focused ADD execution command is:

```sh
GGML_HRX_ENABLE_LOOM=1 GGML_HRX_LOOM_FIRST=1 GGML_HRX_TEST_ONLY=add build-hrx/bin/test-backend-hrx
```

That command exercises route matching, config materialization, embedded Loom source compilation, HSACO emission, HRX executable loading, dispatch, and numeric validation for ADD sizes 1, 257, and 2048. HRX device access may require unsandboxed execution.

## Tests

The route validator checks the catalog schema and route/definition consistency:

```sh
python ggml/src/ggml-hrx/tools/test_validate_loom_routes.py
```

When configured with `GGML_HRX_LOOM=ON`, the hardware-free provider cache test is available through CTest:

```sh
ctest --test-dir build-hrx -R test-hrx-loom-cache --output-on-failure
```

It verifies that identical target, route, source, symbol, and config values produce the same provider cache key, while config value changes and source byte changes produce different keys.

## Current Scope

The current catalog intentionally covers only F32 contiguous same-shape ADD. It does not cover broadcast ADD, non-contiguous ADD, other dtypes, or other operations. Additional Loom routes should be added one route at a time with source, route validation, cache behavior, and execution coverage.
