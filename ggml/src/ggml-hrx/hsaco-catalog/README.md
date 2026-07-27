# HRX HSACO Catalog v0

This catalog is the first HRX direct-dispatch catalog after the removed Loom
catalog path. It uses build-generated HSACO artifacts compiled from existing
CUDA kernel sources.

## Version

- Schema: `ggml-hrx-hsaco-catalog-v0`
- Initial target: `gfx1100`
- Initial op: `GGML_OP_SCALE`
- Initial route: F32 input, F32 output, contiguous source and destination,
  equal shape.

## Source

The first kernel is derived from `ggml/src/ggml-cuda/scale.cu`. The CUDA launch
wrapper and CUDA-only helpers are not copied. The retained device logic is:

- one dimensional grid
- 256 threads per block
- strided loop over `nelements`
- `dst[i] = scale * src[i] + bias`

The standalone HIP source for catalog generation is
`hsaco-catalog/sources/scale_f32.hip`.

## Routing

The v0 runtime route is deliberately hard-coded:

- select `hrx_scale_f32` only for `GGML_OP_SCALE`
- require `GGML_TYPE_F32` source and destination tensors
- require contiguous source and destination tensors
- require source and destination shapes to match
- dispatch with `ceil(nelements / 256)` workgroups and 256 threads

This routing is not the end goal. Later versions can replace it with generated
shape predicates or a richer route table once more kernels exist.

## Integration Checklist

- [x] Add catalog metadata for the initial version, source kernel, and route.
- [x] Generate raw HSACO artifacts at build time.
- [x] Embed the generated HSACO catalog into the HRX backend target.
- [x] Load the selected executable through `hrx_executable_load_data`.
- [x] Route supported tensor shapes to `hrx_stream_dispatch`.
- [x] Add focused backend tests for supported and unsupported `SCALE` cases.
