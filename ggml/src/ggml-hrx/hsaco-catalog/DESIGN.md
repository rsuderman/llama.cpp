# HRX HSACO Catalog Design

The HSACO catalog is the HRX backend's direct-dispatch path for route-specific ggml kernels. Route JSON describes when a kernel applies and how to materialize its invocation. Build tools compile standalone HIP sources to HSACO, embed the artifacts into `ggml-hrx`, and generate C++ route matcher/materializer functions.

## Components

- `metadata.json` declares the catalog schema, version, target architectures, and included route files.
- `defs/*.json` declares compiled kernel ABI: source file, exported symbol, buffer bindings, scalar parameters, constant byte length, and workgroup size.
- `routes/*.json` declares ggml op matching, tensor captures, derived values, kernel binding, scalar packing, and dispatch.
- `sources/*.hip` contains standalone HIP kernels compiled by the catalog generator.
- `tools/validate_hsaco_routes.py` validates metadata, definitions, routes, source references, ABI consistency, source expressions, scalar packing, and dispatch descriptions.
- `tools/generate_hsaco_catalog.py` compiles HIP sources for each target and emits the embedded HSACO catalog table.
- `tools/generate_hsaco_route_impl.py` emits the dispatcher, per-op routers, and per-route matcher/materializer functions.
- `ggml-hrx-hsaco-catalog-runtime.*` owns catalog lifetime, target lookup, lazy executable loading, export ABI validation, tensor binding, scalar dispatch, and `hrx_stream_dispatch`.

## Build Flow

CMake runs two generation paths for `ggml-hrx`.

The catalog artifact path runs `generate_hsaco_catalog.py` over all routes in `metadata.json` and all selected `GGML_HRX_HSACO_TARGETS`. For each route and target, it follows the route `definition`, compiles the definition's HIP source with `hipcc --genco --offload-arch=<target>`, unbundles the AMDGPU object with `clang-offload-bundler`, reads the raw HSACO bytes, and writes generated C++ byte arrays plus catalog entries.

The route implementation path runs `generate_hsaco_route_impl.py` once for the dispatcher and once per operation in `GGML_HRX_HSACO_ROUTE_OPS`. The dispatcher switches on `request->op->op`. Each per-op router sorts candidate routes by descending `priority` and route id, then tries route functions until one returns supported or invoked.

## Runtime Flow

`ggml-hrx.cpp` creates one HSACO catalog per HRX device context on first use. The catalog stores the full HRX architecture string and a target key made by removing any feature suffix after `:`. The target key must match an embedded entry target such as `gfx1100`.

Support queries call `ggml_backend_hrx_hsaco_supports_op()` with no stream and no tensor binder. Generated route code still validates the op, target, tensors, dtypes, attributes, shape captures, derived values, and predicates. If a route matches and no plan is requested, it returns `GGML_BACKEND_HRX_HSACO_INVOKED` with the route id.

Graph compute calls `ggml_backend_hrx_hsaco_invoke()` with a stream and tensor binder. The generated route code fills an execution plan containing the catalog entry, dispatch config, HRX buffer references, packed constants, binding count, and constants size. The runtime lazily loads the selected HSACO, looks up the export symbol, validates export ABI against the catalog entry, caches the loaded route, and dispatches it.

## Routing Generation

Generated route functions follow this sequence:

1. Validate request and ggml op.
2. Find the embedded catalog entry for the route id and current target.
3. Bind tensor role locals such as `src0`, `src1`, and `dst`.
4. Check required tensor presence and declared dtypes.
5. Load declared op attributes.
6. Capture declared tensor shape dimensions.
7. Compute `derived` values in JSON order.
8. Evaluate `match.predicates`.
9. Return supported for support queries.
10. Bind tensors, pack scalar constants, build dispatch config, and return supported for invocation.

Predicate failures map to unsupported reasons: `contiguous` maps to layout, dtype checks map to dtype, target lookup maps to target, and shape predicates map to shape.

## Route Format

Route files use schema `ggml-hrx-hsaco-route-v1` and format `amdgpu-hsaco`.

- `match.tensors` declares tensor roles, dtypes, optional source presence, and optional tensor-local shape captures.
- `match.attributes` declares ggml op attributes used by predicates, derived values, or scalar materialization.
- `match.predicates` declares layout, same-shape, optional-source, and numeric constraints.
- `derived` declares computed scalar values.
- `invocation.buffers` maps tensors to definition bindings.
- `invocation.scalars` maps sources or literals to definition scalar parameters.
- `invocation.dispatch` declares the dispatch shape.

Source strings can refer to:

- `tensor.<role>.type`
- `tensor.<role>.rank`
- `tensor.<role>.element_count`
- `tensor.<role>.dimensions` or `tensor.<role>.dimensions.<index>`
- `tensor.<role>.strides` or `tensor.<role>.strides.<index>`
- `attribute.<name>`
- `shape.<role>.<name>`
- `derived.<name>`

## Shape Capture

Shape capture is declared as an ordered list on a tensor:

```json
"src0": {"type": "F32", "shape": ["ncols", "nrows_0", "nrows_1", "nrows_2"]}
```

The capture at index 0 reads `tensor.src0.dimensions.0`, index 1 reads `tensor.src0.dimensions.1`, and so on. Captured values are referenced through the `shape` namespace:

```json
{"field": "shape.src0.ncols", "equals": 128}
```

Tensor shape capture only names dimensions. Computation over those dimensions belongs in `derived`:

```json
"derived": {
  "nrows": {"type": "i64", "product": ["shape.src0.nrows_0", "shape.src0.nrows_1", "shape.src0.nrows_2"]}
}
```

The generator emits captured shape locals before derived values, so derived expressions, predicates, scalars, and dispatch descriptions can all reference `shape.<role>.<name>`.

## Validation

Validate the catalog with:

```sh
python ggml/src/ggml-hrx/tools/validate_hsaco_routes.py --source-root ggml/src/ggml-hrx/hsaco-catalog
```

Generate one op router for inspection with:

```sh
python ggml/src/ggml-hrx/tools/generate_hsaco_route_impl.py --source-root ggml/src/ggml-hrx/hsaco-catalog --route routes/add_f32.json --out /tmp/add_router.cpp --op-router --op GGML_OP_ADD
```
