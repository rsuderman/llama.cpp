# HRX HSACO Catalog v0

This catalog is the first HRX direct-dispatch catalog after the removed Loom
catalog path. It uses build-generated HSACO artifacts compiled from existing
CUDA kernel sources.

## Version

- Schema: `ggml-hrx-hsaco-catalog-v0`
- Initial target set: `gfx1100`
- Initial ops: `GGML_OP_SCALE`, `GGML_OP_CLAMP`, `GGML_OP_ADD`
- Initial routes: F32 input, F32 output, contiguous source and destination,
  equal shape.

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

## Routing And Artifacts

Routes are logical and target independent. The generator compiles every route
for each platform listed by `GGML_HRX_HSACO_TARGETS` and embeds one HSACO
artifact per route and platform.

The v0 runtime predicates are deliberately simple:

- select `hrx_scale_f32` for `GGML_OP_SCALE`
- select `hrx_clamp_f32` for `GGML_OP_CLAMP`
- select `hrx_add_f32` for `GGML_OP_ADD`
- require `GGML_TYPE_F32` source and destination tensors
- require contiguous source and destination tensors
- require source and destination shapes to match

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
