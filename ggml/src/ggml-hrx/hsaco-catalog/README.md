# HRX HSACO Catalog

See [DESIGN.md](DESIGN.md) for the catalog design, route generation flow, and JSON file format notes.

This catalog is the first HRX direct-dispatch catalog after the removed Loom
catalog path. It uses build-generated HSACO artifacts compiled from existing
CUDA kernel sources.

## Version

- Catalog metadata schema: `ggml-hrx-hsaco-catalog-v0`
- Route schema: `ggml-hrx-hsaco-route-v1`
- Initial target set: `gfx1100`
- Initial ops: `GGML_OP_SCALE`, `GGML_OP_CLAMP`, `GGML_OP_ADD`,
  `GGML_OP_MUL`, `GGML_OP_DIV`, `GGML_OP_SUM_ROWS`, `GGML_OP_SOFT_MAX`,
  `GGML_OP_ARGSORT`
- Initial routes: narrow F32 contiguous pointwise, row-reduction, row-softmax,
  and small row-argsort cases.

## Source

The first kernels are derived from `ggml/src/ggml-cuda/scale.cu`,
`ggml/src/ggml-cuda/clamp.cu`, and `ggml/src/ggml-cuda/binbcast.cu`. The CUDA
launch wrappers and CUDA-only helpers are not copied. The retained `SCALE`
device logic is:

- one dimensional grid
- 256 threads per block
- strided loop over `nelements`
- `dst[i] = scale * src[i] + bias`

The standalone HIP sources for catalog generation are:

- `hsaco-catalog/sources/scale_f32.hip`
- `hsaco-catalog/sources/clamp_f32.hip`
- `hsaco-catalog/sources/add_f32.hip`
- `hsaco-catalog/sources/mul_f32.hip`
- `hsaco-catalog/sources/div_f32.hip`
- `hsaco-catalog/sources/sum_rows_f32.hip`
- `hsaco-catalog/sources/soft_max_f32.hip`
- `hsaco-catalog/sources/argsort_f32_i32.hip`

## Routing And Artifacts

Routes are logical and target independent. The generator compiles every route
for each platform listed by `GGML_HRX_HSACO_TARGETS` and embeds one HSACO
artifact per route and platform.

Definitions in `defs/*.json` describe compiled kernels and ABI: source path,
symbol, bindings, scalar parameters, constant byte length, and fixed workgroup
size.

Routes in `routes/*.json` describe how to use one definition for a ggml op:

- `match` declares the op, tensor dtypes, attributes, and predicates.
- `derived` declares typed values computed from tensor fields, attributes,
  literals, or earlier derived values.
- `invocation` maps tensors and scalars to the kernel ABI and declares dispatch.

Layout, same-shape, optional-source, and numeric constraints belong in
`match.predicates`. Reusable tensor shape values can be named with
`match.tensors.<role>.shape`.

The current runtime predicates are deliberately simple:

- select `hrx_scale_f32` for `GGML_OP_SCALE`
- select `hrx_clamp_f32` for `GGML_OP_CLAMP`
- select `hrx_add_f32` for `GGML_OP_ADD`
- select `hrx_mul_f32` for `GGML_OP_MUL`
- select `hrx_div_f32` for `GGML_OP_DIV`
- select `hrx_sum_rows_f32` for `GGML_OP_SUM_ROWS`
- select `hrx_soft_max_f32` for `GGML_OP_SOFT_MAX`
- select `hrx_argsort_f32_i32` for `GGML_OP_ARGSORT`
- require `GGML_TYPE_F32` source and destination tensors
- require contiguous source and destination tensors
- require source and destination shapes to match, except `MUL` and `DIV`
  support only the model-observed source1 broadcast shapes
- limit `SUM_ROWS`, `SOFT_MAX`, and `ARGSORT` to model-observed column counts

`ggml-hrx.cpp` selects a logical route and packs tensor bindings. The
HSACO-catalog runtime owns target lookup, executable loading, export ABI
validation, dispatch config construction, and the final `hrx_stream_dispatch`
call.

This routing is not the end goal. The route JSON now contains enough metadata
to generate explicit route matcher/materializer functions, but the transitional
top-level `routing` and `launch` fields remain until generic generated
invocation replaces the handwritten route helpers.

## Validation

Validate the catalog metadata, definitions, routes, predicates, derived values,
ABI mappings, scalar packing, and dispatch sources with:

```sh
python ggml/src/ggml-hrx/tools/validate_hsaco_routes.py --source-root ggml/src/ggml-hrx/hsaco-catalog
```

Generate a C++ router implementation for one operation with:

```sh
python ggml/src/ggml-hrx/tools/generate_hsaco_route_impl.py --source-root ggml/src/ggml-hrx/hsaco-catalog --route routes/add_f32.json --out /tmp/add_router.cpp --op-router --op GGML_OP_ADD
```

## Integration Checklist

- [x] Add catalog metadata for the initial version, source kernel, and route.
- [x] Generate raw HSACO artifacts at build time.
- [x] Add an explicit CMake target for HSACO catalog generation.
- [x] Generate platform-specific artifacts from generic route definitions.
- [x] Embed the generated HSACO catalog into the HRX backend target.
- [x] Hide executable loading and dispatch behind the HSACO-catalog runtime.
- [x] Add focused backend tests for supported and unsupported route cases.
