# Apple Metal Framework: How l26f Uses It

## Overview

Metal is Apple's GPU compute and graphics framework. In l26f, we use the **compute**
side only (no graphics). The C host code orchestrates inference; Metal kernels execute
the heavy linear algebra (matvec, attention, normalization, MoE routing) on the GPU.

---

## 1. What Gets Compiled When

### C compile time (`make`)

- All `.c` and `.m` files are compiled by clang/clang++ into a native Mach-O binary
- `.metal` files are **NOT** compiled at this stage — they are plain text files on disk
- The binary contains Objective-C code that will read `.metal` files at runtime

### Runtime (program startup, `l26f_metal_init`)

1. **Source loading**: `ds4_metal_full_source()` reads `metal/preamble.metal` and all
   26 kernel `.metal` files from disk, concatenates them into one big NSString

2. **Library compilation**: `[g_device newLibraryWithSource:source options:options error:&error]`
   sends the concatenated source to Apple's **runtime Metal compiler** (JIT compilation).
   This produces an `MTLLibrary` object — the compiled GPU program.

3. **Function lookup**: For each kernel, `[library newFunctionWithName:@"kernel_name"]`
   retrieves the `MTLFunction` handle. If the runtime compiler silently dropped a
   kernel (e.g., due to unsupported template parameters), this returns `nil`.

4. **Pipeline creation**: `[g_device newComputePipelineStateWithFunction:fn error:&error]`
   creates a compiled, optimized pipeline state object (PSO). This is the expensive
   step — it involves GPU driver compilation. The PSO is cached for the lifetime of
   the program.

### Per-inference-step

- No compilation happens during inference. All PSOs are pre-built at init.
- Kernel dispatch is: get encoder → set pipeline → bind buffers → dispatch → end encoding.

---

## 2. The Runtime vs Offline Compiler Gap

**This is the single most important thing to know.**

Apple provides two Metal compilers:

| Compiler | How | Capabilities |
|----------|-----|-------------|
| **Offline** (`xcrun -sdk macosx metal`) | Build-time, produces `.metallib` files | Full C++14/Metal feature set, templates, function pointers |
| **Runtime** (`newLibraryWithSource:`) | Runtime JIT from source string | Subset of Metal features; silently drops unsupported kernels |

The **runtime compiler** (which l26f uses) has a critical limitation: **it silently
drops kernel functions that use function-pointer template parameters.** No error,
no warning — the function simply doesn't appear in the library's `functionNames`.
`newFunctionWithName:` returns `nil`.

This is why `tests/check_metal.sh` exists — it compiles with the offline compiler to
catch syntax errors, but **cannot** catch features unsupported by the runtime compiler.

### What works in runtime compilation

- Concrete (non-template) `kernel void` functions
- `struct`, `enum`, `typedef`, `#define`, `constant` arrays
- `static inline` helper functions
- `thread`, `threadgroup`, `device`, `constant` address spaces
- SIMDgroup operations (`simdgroup_matrix`, `simd_broadcast`)
- `_Static_assert`

### What does NOT work in runtime compilation

- Function-pointer template parameters: `template<typename... F> kernel void foo(F... funcs)`
- Some advanced C++ features (SFINAE, complex template metaprogramming)

### Prevention pattern

After writing a new Metal kernel, always add a load-time check:

```objc
id<MTLFunction> f = [library newFunctionWithName:@"kernel_my_new_kernel"];
if (!f) {
    fprintf(stderr, "l26f: MISSING kernel: kernel_my_new_kernel\n");
    // ... cleanup and fail
}
```

---

## 3. Object Lifecycle

```
MTLDevice          g_device       — Singleton GPU device, created once
MTLCommandQueue    g_queue        — Singleton command queue, created once
MTLLibrary         g_library      — Compiled shader library, created once at init
MTLComputePipelineState  g_*_pipeline  — Per-kernel PSO, created once at init
MTLCommandBuffer   per-dispatch   — Created per batch of kernel dispatches
MTLComputeCommandEncoder  per-kernel — Created per kernel dispatch, lightweight
MTLBuffer          ds4_metal_tensor — GPU memory allocation, lives across dispatches
```

### Tensor lifecycle

`ds4_metal_tensor` wraps an `MTLBuffer` (GPU memory). Key properties:

- **Storage mode**: `MTLResourceStorageModeShared` — CPU and GPU share the same
  memory. No explicit copy needed, but **memory is NOT zeroed on allocation**.
  Uninitialized reads produce garbage that propagates through the entire computation.
  Always zero-fill after allocation (manifesto rule #1).

- **Creation**: `ds4_metal_tensor_alloc(bytes)` allocates GPU-visible memory.
- **Write**: `ds4_metal_tensor_write(tensor, offset, cpu_data, bytes)` — memcpy from
  CPU to shared memory.
- **Read**: `ds4_metal_tensor_read(tensor, offset, cpu_buf, bytes)` — memcpy from
  shared memory to CPU.
- **Model weights**: Zero-copy via `MTLBuffer` created from mmap'd model file using
  `newBufferWithBytesNoCopy:length:options:`. The GPU reads weights directly from the
  mmap'd file pages.

---

## 4. Command Buffer and Dispatch Pattern

The dispatch flow for each inference step:

```
l26f_metal_begin_commands()   → creates MTLCommandBuffer + MTLComputeCommandEncoder
  ├── set pipeline for kernel_1
  ├── bind buffers (setBuffer:offset:atIndex:)
  ├── dispatch (dispatchThreadgroups:threadsPerThreadgroup:)
  ├── set pipeline for kernel_2
  ├── bind buffers
  ├── dispatch
  └── ...
l26f_metal_end_commands()     → endEncoding + commit command buffer
l26f_metal_synchronize()      → wait for GPU to finish (optional, for profiling)
```

Multiple kernels can be encoded into one command buffer — they execute sequentially
on the GPU. The GPU starts executing as soon as the command buffer is committed.

### Threadgroup sizing

Metal dispatches work in two levels:

- **Grid** (total work): `MTLSizeMake(total_elements, 1, 1)`
- **Threadgroup** (per-SM block): `MTLSizeMake(256, 1, 1)` typical

The GPU launches `ceil(grid / threadgroup)` threadgroups. Each threadgroup runs
up to 256 threads (SIMD width = 32 on Apple Silicon, 8 SIMD lanes per threadgroup).

---

## 5. Source File Organization

```
metal/preamble.metal       — Base includes, defines, quant block structs, lookup tables
metal/*.metal              — Kernel implementations (26 files)
l26f_metal.m               — Objective-C glue: device init, library compilation,
                              pipeline creation, kernel dispatch functions
l26f_metal.h               — C declarations for dispatch functions (no ObjC types)
tests/check_metal.sh       — Offline compilation test for all .metal sources
```

### Adding a new kernel

1. Write the kernel in `metal/something.metal` (or add to an existing file)
2. Add the file to `required_sources` in `ds4_metal_full_source()` in `l26f_metal.m`
3. Add pipeline creation in `l26f_metal_init()`:
   ```objc
   fn = [library newFunctionWithName:@"kernel_my_thing"];
   if (!fn) { fprintf(stderr, "..."); return 0; }
   g_my_thing_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
   ```
4. Write a C dispatch function in `l26f_metal.m` that sets up and dispatches the kernel
5. Declare it in `l26f_metal.h`
6. Run `tests/check_metal.sh` to verify offline compilation
7. Run `make debug && build_debug/test_l26f_multilayer ...` to verify runtime compilation

---

## 6. Debugging Metal Kernels

### Kernel not found (nil function)

1. Run `tests/check_metal.sh` — if offline compilation fails, fix syntax
2. If offline passes but runtime fails: you're using a feature the runtime compiler
   doesn't support (function-pointer templates are the most common)
3. Check the startup validation in `l26f_metal_init()` for missing kernel warnings

### Wrong results

1. Check buffer sizes and offsets — off-by-one in threadgroup dispatch leaves
   trailing elements unwritten
2. Check that all GPU buffers are zeroed after allocation (manifesto rule #1)
3. Use checksums to pinpoint which kernel produces wrong output (manifesto rule #2)
4. Read back tensors to CPU and print: `ds4_metal_tensor_read` + `fprintf`

### Performance

1. Profile with `L26F_PROFILE=1 L26F_PROFILE_SYNC=1` environment variables
2. Profile breakdown: MoE ~50%, GLA ~37%, MLA ~7%, Logits ~5%
3. Batch dispatch reduces overhead: `l26f_metal_begin_commands()` before multiple
   kernel calls, `l26f_metal_end_commands()` after
