# HRX Loom Catalog

This catalog contains Loom routes for HRX. Routes, definitions, and sources are grouped by platform folder before operation, dtype, and route name. The current checked-in routes use `generic` because they are platform agnostic.

## Version

- Catalog metadata schema: `ggml-hrx-loom-catalog-v0`
- Definition schema: `ggml-hrx-loom-def-v1`
- Route schema: `ggml-hrx-loom-route-v1`
- Initial target set: `gfx1100`
- Platform folders: `generic`

## Files

- `metadata.json` lists the catalog version, target set, and route files.
- `defs/generic/<op>/<dtype>/<name>.json` describes the Loom source identity and runtime ABI.
- `routes/generic/<op>/<dtype>/<name>.json` describes route matching, derived values, compile config, tensor and scalar invocation, and dispatch geometry.
- `sources/generic/<op>/<dtype>/<name>.loom` contains platform-agnostic Loom source.

## Route Shape

Each route captures the tensor dtypes, shapes, layout predicates, derived values, compile config bindings, invocation buffers and scalars, and dispatch geometry needed by one Loom kernel.

This catalog uses only the final Loom route fields: `match`, `derived`, `config`, and `invocation`. It does not include HSACO compatibility fields.

The `config` section is required. Its `mode` must be `compile`, and `bindings` is an ordered array of compile-time values. Each binding has a C identifier `name`, a scalar `type`, and exactly one of `source` or `value`. The current generated runtime stores at most 32 config bindings, with names limited to 63 bytes and formatted values limited to 127 bytes.

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

The Loom catalog is part of the HRX backend build. Configure HRX with both the HRX runtime package and the Loom C API package:

```sh
cmake -S . -B build-hrx -DGGML_HRX=ON \
  -Dhrx_DIR=/path/to/hrx/libhrx/cmake/hrx \
  -Dloomc_DIR=/path/to/hrx/loom/binding/c/cmake/loomc
```

The first supported source format is `loom-text`, which is compiled in-process through `loomc` and emitted as AMDGPU HSACO before loading through HRX. Loom routing is always enabled for HRX; there is no build or runtime switch to disable the catalog.

The focused ADD execution command is:

```sh
GGML_HRX_TEST_ONLY=add build-hrx/bin/test-backend-hrx
```

That command exercises route matching, config materialization, embedded Loom source compilation, HSACO emission, HRX executable loading, dispatch, and numeric validation for ADD sizes 1, 257, and 2048. HRX device access may require unsandboxed execution.

## Tests

The route validator checks the catalog schema and route/definition consistency:

```sh
python ggml/src/ggml-hrx/tools/test_validate_loom_routes.py
```

The hardware-free provider cache test is available through CTest:

```sh
ctest --test-dir build-hrx -R test-hrx-loom-cache --output-on-failure
```

It verifies that identical target, route, source, symbol, and config values produce the same provider cache key, while config value changes and source byte changes produce different keys.

## Current Scope

The current generic Loom catalog covers selected ADD, ARGSORT, CLAMP, CPY, DIV, MUL, RMS_NORM, ROPE, SUB, and SUM_ROWS routes. Platform-specific variants should be added under their own platform folder instead of mixing with `generic`.
