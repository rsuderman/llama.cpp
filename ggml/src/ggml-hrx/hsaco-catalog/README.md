# HRX HSACO Catalog v0

This catalog is the first HRX direct-dispatch catalog after the removed Loom
catalog path. It uses build-generated HSACO artifacts compiled from existing
CUDA kernel sources.

## Version

- Schema: `ggml-hrx-hsaco-catalog-v0`
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

The v0 runtime predicates are deliberately simple:

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

This routing is not the end goal. Later versions can replace it with generated
shape predicates or a richer route table once more kernels exist.

## Integration Checklist

- [x] Add catalog metadata for the initial version, source kernel, and route.
- [x] Generate raw HSACO artifacts at build time.
- [x] Add an explicit CMake target for HSACO catalog generation.
- [x] Generate platform-specific artifacts from generic route definitions.
- [x] Embed the generated HSACO catalog into the HRX backend target.
- [x] Hide executable loading and dispatch behind the HSACO-catalog runtime.
- [x] Add focused backend tests for supported and unsupported route cases.
