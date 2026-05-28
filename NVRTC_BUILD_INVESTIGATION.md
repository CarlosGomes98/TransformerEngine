# NVRTC Fused Softmax — Build Time Investigation Log

**Date:** 2026-05-28
**Branch:** `cgomes/nvrtc-softmax-ci`
**Goal:** Figure out why the recent NVRTC fused-softmax change (commit `0b655b679`)
did not visibly improve TE build times. Produce a breakdown of where build time goes,
and look more carefully at kernel-compilation costs.

## Plan of work

1. Inspect what the commit actually changed (which TUs, which template fanout removed).
2. Run `build_tools/compile_time_report.py` against the existing `.ninja_log`s to see
   current per-TU compile times and identify hotspots.
3. Capture a clean-build baseline (full timing) so we can attribute the total budget.
4. Diff the softmax TU times before vs. after; quantify the share of total build time
   they actually consumed.
5. Look at the rest of the build cost (link, other large CUDA TUs, codegen, Python build
   overhead) to see whether softmax migration could have moved the needle at all.

The log below is appended as findings come in.

## Findings

### 1. What the commit actually does

Commit `0b655b679` ("Add RTC fused softmax build test changes"):

- Adds an RTC compile/launch path inside `scaled_masked_softmax.cu`,
  `scaled_upper_triang_masked_softmax.cu`, and `scaled_aligned_causal_masked_softmax.cu`.
  Each `dispatch_*` template function chooses between `rtc::is_enabled()` →
  `KernelManager::compile` + `launch` and a static `switch` over `log2_elements`
  that calls the original `__global__` template.
- Wraps the static switch in `#if NVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX … #else
  throw_nvrtc_required(…) #endif`. The macro defaults to `0` in the source via
  `#ifndef`.
- Adds a CMake option `NVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX` (default `OFF`) and
  registers the three softmax `.cu` files as string headers (mirroring the existing
  transpose RTC pattern).
- Adds an RTC unit test and a softmax operator test.

So the *intent* is: with the option `OFF`, the static template fanout (15 log₂ ×
2 dtypes × forward/backward) is `#if`-removed inside the `.cu` files, and only the
RTC path is reachable at runtime.

### 2. Why the build didn't get faster — primary cause

The `build/` tree was last fully configured on **2026-02-25**, but the source
changes are from **today (2026-05-28)**:

| Path | mtime |
| --- | --- |
| `transformer_engine/common/fused_softmax/scaled_masked_softmax.cu` | 2026-05-28 14:35 |
| `transformer_engine/common/CMakeLists.txt` | 2026-05-28 14:35 |
| `build/cmake/build.ninja` | 2026-02-25 11:37 |
| `build/cmake/CMakeCache.txt` | 2026-02-25 11:37 |
| `build/cmake/CMakeFiles/.../scaled_masked_softmax.cu.o` | 2026-02-25 11:44 |
| `build/cmake/string_headers/` | only transpose RTC headers present |

So `build/cmake/` is the *old* tree:
- `build.ninja` has not been regenerated since the `CMakeLists.txt` change, so it
  does not produce the new `string_code_fused_softmax_*.h` headers and it does
  not pass `-DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=…` to the compiler. Verified
  with `grep "scaled_masked_softmax.cu.o" build/cmake/build.ninja`:
  the `DEFINES` line is only `-DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
  -Dtransformer_engine_EXPORTS`. No softmax legacy flag.
- The current `.o` files are 3 months old. `nm -C` on
  `scaled_masked_softmax.cu.o` shows **all 15 log₂ × 2 dtype × forward/backward**
  `__device_stub__` entries; `cuobjdump --dump-resource-usage` confirms the full
  kernel set is in the fatbin. That's the pre-change layout.

Confirms that **the static instantiations are still in the .o** — but only because
that .o has not been rebuilt yet, not because the source change is broken.

### 3. Why the build wouldn't have been *dramatically* faster anyway

Top compile-time TUs from the existing `build/cmake/.ninja_log`
(sequential ms, `compile_time_report.py` output):

| Rank | Time (s) | TU |
| --: | --: | --- |
| 1 | 84.7 | `transpose/cast_transpose_fusion.cu` |
| 2 | 72.3 | `normalization/layernorm/ln_fwd_cuda_kernel.cu` |
| 3 | 53.1 | `activation/gelu.cu` |
| 4 | 52.0 | `fused_attn/fused_attn_fp8.cu` |
| 5 | 49.2 | `fused_attn/fused_attn_f16_arbitrary_seqlen.cu` |
| 6 | 48.5 | `normalization/layernorm/ln_bwd_semi_cuda_kernel.cu` |
| 7 | 47.6 | `activation/relu.cu` |
| 8 | 40.9 | `normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu` |
| 9 | 40.3 | `gemm/cutlass_grouped_gemm.cu` |
| 10 | 39.6 | `normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu` |
| 11 | 39.4 | `activation/swiglu.cu` |
| 12 | 37.1 | `fused_attn/fused_attn_f16_max512_seqlen.cu` |
| 13 | 35.3 | `fused_attn/utils.cu` |
| … | … | … |
| 21 | 13.6 | `fused_softmax/scaled_upper_triang_masked_softmax.cu` |
| 23 | 13.1 | `fused_softmax/scaled_masked_softmax.cu` |
| 26 | 11.9 | `fused_softmax/scaled_aligned_causal_masked_softmax.cu` |

Total sequential compile time across the 75 listed outputs: **1043.6 s**. The
three fused-softmax TUs sum to **38.5 s ≈ 3.7%** of total sequential build time.
Even if RTC migration drove them to ~0 s, the wall-clock impact for a parallel
build is small.

More importantly, the actual **wall-clock** is bounded by the longest dependency
chain across `-j`, which is dominated by the 84 s `cast_transpose_fusion.cu`
build and ~70 s normalization TUs — softmax is not on that critical path.

So even a working softmax-RTC change should only have removed a few seconds.

### 4. Confirmation that the change *does* work when actually applied

After re-running cmake on `build/cmake/` (which regenerated `build.ninja` to pass
`-DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=0` and to materialize the
`string_code_fused_softmax_*.h` headers under `build/cmake/string_headers/`), I
rebuilt the three softmax TUs in isolation under both modes:

| Mode | Wall (3 TUs, -j max) | CPU user | Max RSS | Sum of .o sizes |
| --- | --: | --: | --: | --: |
| Legacy static (`NVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=ON`) | 11.95 s | 32.67 s | 324 MB | **6.76 MB** (2.21 + 2.53 + 2.02) |
| NVRTC (`OFF`, default after this MR) | **2.67 s** | 6.98 s | 270 MB | **0.85 MB** (0.27 + 0.31 + 0.28) |

So the three softmax TUs go from ~11 s sequential each (~33 s CPU total) to
~2.3 s sequential each (~7 s CPU total). About **4×** speedup per TU. Object
sizes drop ~8× because the 15 log₂ × 2 dtype × {fwd, bwd} kernel instantiations
are no longer compiled in.

The static instantiation removal is verified by `nm -C` and
`cuobjdump --dump-resource-usage`:
- Legacy `.o`: 15 `__device_stub__` entries per kernel template per dtype, fatbin
  contains all kernels.
- NVRTC `.o`: only the dispatch wrappers and the RTC kernel-source string symbol;
  no `__device_stub__` entries for `scaled_*softmax_warp_*`.

### 5. What the user most likely saw

Putting this together:

1. The MR change is correct, but **CMake was never re-run** after the
   `CMakeLists.txt` edit, so the build kept using the old `build.ninja`. That
   `build.ninja` (a) did not define `NVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX` and
   (b) did not produce the new `string_code_*.h` headers.
2. Without the macro definition, the in-source `#ifndef … #define … 0` default
   should *still* take effect. So a fresh rebuild of the `.cu` files should
   trigger NVRTC mode just from the source change. But the user's ninja
   invocation may not have rebuilt anything — ninja considered the existing
   `.o` files current under the old build.ninja, and the rebuild of the `.cu`
   would have failed at compile time anyway because the include of
   `string_code_fused_softmax_*_cu.h` would be missing.
3. Either way, the user observed "build time did not change," which is exactly
   what a stale build tree with no rebuild looks like.

**Action for the user:** delete `build/cmake/CMakeCache.txt` (or the whole
`build/cmake/` directory) and re-run `pip install -e . --no-build-isolation`
(or `python setup.py build_ext`), so cmake regenerates `build.ninja` with the
new options and generated headers, then ninja rebuilds the affected TUs.

### 6. gcc 13 ICE on `transpose/cast_transpose_fusion.cu` (pre-existing, unrelated)

The first full rebuild attempt under gcc 13 (system default) failed with an
internal compiler error:

```
during RTL pass: cprop
…/util/rtc.h: In member function ‘void transformer_engine::rtc::KernelManager::launch(
    const std::string&, dim3, dim3, unsigned int, cudaStream_t, ArgTs&& ...)
    [with ArgTs = {transformer_engine::CTDBiasDActParam<…>&, …}]’:
…/util/rtc.h:150:1: internal compiler error: in try_forward_edges, at cfgcleanup.cc:580
```

- gcc version: `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`.
- The ICE is reproducible when compiling `cast_transpose_fusion.cu` in
  isolation; it is **not** caused by this MR. The crash is inside
  `KernelManager::launch<…>` template instantiation for a `CTDBiasDActParam<…>`
  argument list, which is exercised by the *existing* transpose RTC code.
- Workaround verified: switching the CUDA host compiler to `g++-12`
  (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`) lets the TU build.
- This is worth noting because it can derail the build right after the change
  is properly configured, making it easy to misread the failure as caused by
  the new softmax code.

### 7. Full-build breakdown — current branch, NVRTC mode, gcc-12 host

A clean full build of `libtransformer_engine.so` (the only target affected by
this MR; PyTorch/JAX wrappers in `build/temp.linux-x86_64-cpython-312/`
are independent) on this 32-core machine:

```
wall    = 110.89 s
cpu_user = 1624.70 s   # ≈ sequential equivalent of compile work
cpu_sys  =   71.75 s
max_rss  = 4063 MB     # peak across all parallel nvcc jobs
outputs  = 79 TUs + 1 link
sequential compile total (from .ninja_log) = 1797.1 s
effective parallelism = 1797.1 / 110.89 = 16.2× (50.6% of 32 cores)
```

Parallelism falls off sharply as the build runs because only a handful of
long-pole TUs survive past ~70 s:

| t (s) | concurrent nvcc jobs |
| --: | --: |
| 10 | 34 |
| 20 | 34 |
| 30 | 22 |
| 40 | 15 |
| 50 | 15 |
| 60 | 13 |
| 70 | 11 |
| 80 | 6 |
| 90 | 5 |
| 100 | 2 |
| 110 | 1 |

The 0.4 s link at the end is dominated by waiting on the last compile to
finish.

#### Wall-clock critical path

The 10 TUs that determine total wall-clock (all > 70 s, all live in the
build's tail):

| End at (s) | Elapsed (s) | TU |
| --: | --: | --- |
| 110.5 | 103.3 | `normalization/layernorm/ln_fwd_cuda_kernel.cu` |
| 103.5 | 103.5 | `transpose/cast_transpose_fusion.cu` |
|  93.9 |  93.9 | `activation/relu.cu` |
|  93.6 |  87.6 | `fused_attn/fused_attn_fp8.cu` |
|  91.7 |  91.7 | `activation/gelu.cu` |
|  89.6 |  84.1 | `fused_attn/fused_attn_f16_arbitrary_seqlen.cu` |
|  80.0 |  72.8 | `normalization/layernorm/ln_bwd_semi_cuda_kernel.cu` |
|  79.7 |  61.5 | `fused_router/fused_topk_with_score_function.cu` |
|  74.3 |  66.9 | `normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu` |
|  74.1 |  66.8 | `normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu` |

#### Where the fused-softmax TUs land now (post-MR)

| TU | Time (s) |
| --- | --: |
| `fused_softmax/scaled_aligned_causal_masked_softmax.cu.o` | 7.4 |
| `fused_softmax/scaled_masked_softmax.cu.o` | 6.4 |
| `fused_softmax/scaled_upper_triang_masked_softmax.cu.o` | 6.0 |
| `util/rtc.cpp.o` | 3.3 |
| **Sum of softmax TUs (post-MR)** | **19.8 s** |
| Sum of softmax TUs (pre-MR, from old `.ninja_log`) | ~38.6 s |
| Sequential saving from this MR | ~18.8 s (about 1.0% of total sequential build) |
| Wall-clock saving on this machine | ~0 s |

The softmax TUs finish in the first ~25 s of the build — during the
high-concurrency phase where 22–34 jobs are running in parallel. They are not
on the critical path. Even if the MR drove them to literally 0 s, the build
wall-clock would not move, because the longest TU (`cast_transpose_fusion.cu`,
103.5 s) starts at t=0 and runs to t=103.5 regardless.

## Summary — answer to "why didn't build time improve?"

**Two reasons, in order of impact:**

1. **The MR was never built.** `build/cmake/` was last configured 3 months
   ago (2026-02-25). The CMake change on `2026-05-28` was never picked up:
   `build.ninja` did not pass the new `-DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=0`
   flag and the new `string_code_fused_softmax_*.h` headers were never
   generated. The existing `.o` files (and the per-TU times in `.ninja_log`)
   reflect the pre-MR layout, complete with all 15 log₂ × 2 dtype × {fwd, bwd}
   kernel instantiations. To pick up the change you need a CMake reconfigure —
   `rm -rf build/cmake && pip install -e . --no-build-isolation` (or just
   `rm build/cmake/CMakeCache.txt && python setup.py build_ext`).
2. **Even when applied, the softmax MR is not on the wall-clock critical path.**
   The three softmax TUs go from ~38.6 s to ~19.8 s sequential (about 4× per
   TU, confirmed by direct A/B with `NVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=ON`
   vs `OFF`), but the build wall-clock is set by `cast_transpose_fusion.cu`
   (104 s) and `ln_fwd_cuda_kernel.cu` (103 s). Softmax compiles entirely
   inside the first 25 s, in parallel with 22–34 other jobs, so saving 19 s of
   sequential work shaves ~0 s of wall-clock.

The MR is correct and worth keeping — it removes ~18.8 s of sequential CPU
work and ~6 MB of object code per build, which compounds with later phases of
the plan. But on its own, on this machine, it cannot move wall-clock.

## Recommendations for measurable build-time wins

Any change targeting wall-clock must move work *off the long-pole TUs in the
critical-path band* (the 10 TUs above, all > 60 s). The plan's prioritization
matches this:

1. **Phase 2 (Normalization RTC)** — `ln_fwd_cuda_kernel.cu` (103 s),
   `ln_bwd_semi_cuda_kernel.cu` (73 s), `rmsnorm_fwd_cuda_kernel.cu` (67 s),
   `rmsnorm_bwd_semi_cuda_kernel.cu` (67 s). Four TUs at the top of the
   critical-path band. Highest expected wall-clock impact.
2. **Phase 3 (Activation + FP8/MXFP8 quantize)** — `relu.cu` (94 s),
   `gelu.cu` (92 s), `swiglu.cu` (71 s). All on the critical-path band.
3. **`fused_attn_fp8.cu` (88 s)** and `fused_attn_f16_arbitrary_seqlen.cu`
   (84 s). Not in the plan as separate phases yet; worth measuring once Phase 2
   / 3 are done to see what new bottleneck appears.
4. **`cast_transpose_fusion.cu` (104 s)** already uses RTC for the no-bias
   path; remaining cost is the static `CTDBiasDActParam<…>` template fanout.
   Worth a focused look — this is also where gcc 13 ICEs.

For reference, the toolchain on this host is `gcc 13.3.0` + `nvcc 13.1` and
needs `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12` to avoid a pre-existing
gcc-13 ICE on `cast_transpose_fusion.cu`.

## Phase 2 experiment: normalization static-fanout removal

Goal: measure the upper bound of what an NVRTC migration of the LayerNorm /
RMSNorm forward+backward kernels could save. Mirrors the softmax `#if`-gating
pattern but does **not** yet wire an NVRTC dispatch backend — when the flag is
flipped OFF the runtime will fail (`Unavailable kernel for this normalization
config`) until the real Phase 2 lands. That is fine for build-time
measurement.

### Change

Added a new CMake option `NVTE_BUILD_LEGACY_STATIC_NORM` (default `ON`) and
wrapped the `REGISTER_NORM_LAUNCHER(...)` blocks in all four TUs with
`#if NVTE_BUILD_LEGACY_STATIC_NORM`. With `ON`, behavior is unchanged from
main. With `OFF`, the file compiles to the headers + un-instantiated launcher
templates only.

Files touched:

- `transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu`
- `transformer_engine/common/normalization/layernorm/ln_bwd_semi_cuda_kernel.cu`
- `transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu`
- `transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu`
- `transformer_engine/common/CMakeLists.txt`

The four `.cu` files together register **241 + 151 + 77 + 87 = 556**
`REGISTER_NORM_LAUNCHER` template instantiations across hidden-size × dtype
combinations.

### A/B per-TU measurement (4 TUs built in parallel, gcc-12 host)

| Mode | Wall | CPU user | Max RSS | ln_fwd .o | ln_bwd .o | rmsnorm_fwd .o | rmsnorm_bwd .o | Sum .o |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| Legacy `ON` (current) | **67.12 s** | 186.22 s | 3.87 GB | 9.68 MB | 5.86 MB | 2.69 MB | 2.81 MB | **21.04 MB** |
| Legacy `OFF` (floor) | **30.26 s** | 113.47 s | 3.72 GB | 111 KB | 111 KB | 111 KB | 111 KB | **0.44 MB** |
| Δ | **−36.86 s (−55%)** | **−72.75 s (−39%)** | | | | | | **−98%** |

The floor `.o` files are all 111 KB because the launcher templates are no
longer instantiated; the file compiles to a small set of header-driven
symbols. The remaining ~28 s of per-TU work is the cost of parsing the
norm-common + cudnn-frontend + cutlass headers and checking the un-instantiated
launcher template definitions.

### A/B full-build wall-clock (single `ninja transformer_engine`, clean)

| Mode | Wall | CPU user | Sequential total | Parallelism | ln_fwd | ln_bwd | rmsnorm_fwd | rmsnorm_bwd | Sum norm |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Legacy `ON` (current) | **110.89 s** | 1624.70 s | 1797.1 s | 16.21× | 103.3 s | 72.8 s | 66.9 s | 66.8 s | **309.8 s** |
| Legacy `OFF` (floor) | **104.07 s** | 1554.75 s | 1739.7 s | 16.72× | 53.9 s | 59.0 s | 55.1 s | 56.6 s | **224.6 s** |
| Δ | **−6.82 s (−6.2%)** | **−69.95 s (−4.3%)** | −57.4 s | | −49.4 s | −13.8 s | −11.8 s | −10.2 s | **−85.2 s** |

### Why per-TU saves 49 s but wall-clock only saves 7 s

The same critical-path dynamic from the softmax investigation, just with
larger numbers. Tail of the build under `legacy OFF`:

| End at (s) | TU |
| --: | --- |
| 103.7 | `transpose/cast_transpose_fusion.cu` |
|  97.5 | `activation/gelu.cu` |
|  89.1 | `fused_attn/fused_attn_f16_arbitrary_seqlen.cu` |
|  88.7 | `fused_attn/fused_attn_fp8.cu` |
|  88.1 | `activation/relu.cu` |
|  73.1 | `fused_router/fused_topk_with_score_function.cu` |
|  72.0 | `gemm/cutlass_grouped_gemm.cu` |
|  68.4 | `activation/swiglu.cu` |
|  67.8 | `normalization/common.cpp` |
|  66.4 | `fused_attn/utils.cu` |
|  65.9 | `normalization/layernorm/ln_bwd_semi_cuda_kernel.cu` |

Under `legacy ON` the critical path was `ln_fwd_cuda_kernel.cu` at 110.5 s.
Removing its static fanout collapses it to 53.9 s — but the next-longest TU,
`transpose/cast_transpose_fusion.cu` at 103.7 s, becomes the new ceiling. So
the wall-clock saving is bounded by the *gap* between the old #1 and the new
#1 critical-path TU (110.5 − 103.7 ≈ 7 s), not by the per-TU savings.

### Headroom past this measurement

The 53.9 s floor on `ln_fwd_cuda_kernel.cu` is what a full Phase 2 migration
would inherit if it only stops at "no static instantiations". To go below
that floor, the real Phase 2 MR also needs to:

1. **Trim the host-side TU's include footprint.** Most of the remaining 53.9 s
   is spent parsing `cudnn_frontend`, `cutlass`, and the kernel headers — none
   of which are needed by a dispatch-only host file. Splitting the registry/
   dispatch into a separate `.cpp` that doesn't include the kernel headers,
   and embedding the kernel device code as an RTC source-string header (the
   softmax pattern), should bring this down to a few seconds, in line with
   the current softmax TUs.
2. **Make the OFF mode functional** by routing the empty registry through an
   NVRTC `KernelManager` dispatch keyed on (norm type, stage, dtypes, hidden
   size, ...).
3. Preserve `cudaOccupancyMaxActiveBlocksPerMultiprocessor` semantics with
   `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the `CUfunction` returned
   by NVRTC.

With those done, the same 4 TUs should drop from ~67 s wall (4-in-parallel
legacy) to roughly the softmax range (~3 s wall, 4 in parallel), and the
critical-path-bounded full-build saving would shift to whatever the next
ceiling becomes (`cast_transpose_fusion.cu` at ~104 s today).

### Summary of what this experiment buys

- Verified that **the upper bound for a Phase 2 NVRTC migration of norm is
  ~7 s wall-clock + ~70 s sequential CPU** on this hardware.
- Identified that the next ceiling after norm is `cast_transpose_fusion.cu`,
  not norm. Phase 2 alone will not get the build under ~100 s; doing Phase 3
  (activation/quantize: relu, gelu, swiglu are all 68–98 s) is what would.
- The 21 MB → 0.44 MB object-size reduction translates directly into smaller
  `libtransformer_engine.so` link cost (not significant on this machine, but
  noticeable on slower disks/CI runners).
- The `NVTE_BUILD_LEGACY_STATIC_NORM` flag (default `ON`) is a non-invasive
  scaffold for the real Phase 2 MR to flip the default once the RTC backend
  is in place.

### Reproduction

```bash
# Reconfigure with the new flag (default ON = current behavior)
cmake -DNVTE_BUILD_LEGACY_STATIC_NORM=ON build/cmake

# A/B per-TU (4 norm TUs)
rm -f build/cmake/CMakeFiles/transformer_engine.dir/normalization/{layernorm,rmsnorm}/*.o
/usr/bin/time .venv/bin/ninja \
  CMakeFiles/transformer_engine.dir/normalization/layernorm/ln_fwd_cuda_kernel.cu.o \
  CMakeFiles/transformer_engine.dir/normalization/layernorm/ln_bwd_semi_cuda_kernel.cu.o \
  CMakeFiles/transformer_engine.dir/normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu.o \
  CMakeFiles/transformer_engine.dir/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu.o
# → wall 67.12 s, .o sum 21 MB

cmake -DNVTE_BUILD_LEGACY_STATIC_NORM=OFF build/cmake
rm -f build/cmake/CMakeFiles/transformer_engine.dir/normalization/{layernorm,rmsnorm}/*.o
/usr/bin/time .venv/bin/ninja [same four targets]
# → wall 30.26 s, .o sum 0.44 MB

# Full-build A/B
cmake -DNVTE_BUILD_LEGACY_STATIC_NORM=ON build/cmake
find build/cmake/CMakeFiles/transformer_engine.dir -name '*.o' -delete
rm -f build/cmake/.ninja_log
/usr/bin/time .venv/bin/ninja transformer_engine     # → 110.89 s
cmake -DNVTE_BUILD_LEGACY_STATIC_NORM=OFF build/cmake
find build/cmake/CMakeFiles/transformer_engine.dir -name '*.o' -delete
rm -f build/cmake/.ninja_log
/usr/bin/time .venv/bin/ninja transformer_engine     # → 104.07 s
```

## Reproduction commands used

```bash
# Save baseline (pre-MR-effect ninja log, stale from Feb 25)
cp build/cmake/.ninja_log .ninja_log.before-rebuild

# Reconfigure cmake so the MR actually takes effect
rm -f build/cmake/CMakeCache.txt build/cmake/.ninja_log build/cmake/.ninja_deps
find build/cmake/CMakeFiles/transformer_engine.dir -name '*.o' -delete
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake \
  -S transformer_engine/common -B build/cmake \
  -DPython_EXECUTABLE=.venv/bin/python \
  -DPython_INCLUDE_DIR=/usr/include/python3.12 \
  -DPython_SITEARCH=.venv/lib/python3.12/site-packages \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja \
  -Dpybind11_DIR=.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11 \
  -GNinja

# Full timed build of libtransformer_engine.so
/usr/bin/time -f "wall=%es cpu_user=%Us cpu_sys=%Ss max_rss=%MKB" \
  .venv/bin/ninja transformer_engine

# A/B per-TU softmax measurement
cmake -DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=ON build/cmake
rm build/cmake/CMakeFiles/transformer_engine.dir/fused_softmax/*.o
/usr/bin/time .venv/bin/ninja \
  CMakeFiles/transformer_engine.dir/fused_softmax/scaled_masked_softmax.cu.o \
  CMakeFiles/transformer_engine.dir/fused_softmax/scaled_upper_triang_masked_softmax.cu.o \
  CMakeFiles/transformer_engine.dir/fused_softmax/scaled_aligned_causal_masked_softmax.cu.o
# repeat with -DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=OFF

# Per-TU report
python3 build_tools/compile_time_report.py \
  --ninja-log build/cmake/.ninja_log --limit 25
```

## Session 2026-05-28 (cont.): validation container + Phase 2 norm RTC bring-up

### Environment

Moved to a new validation container. Differs from the original workstation:

- **CUDA 12.8** (`/usr/local/cuda`, `nvcc 12.8.93`), not 13.1.
- Host compiler **g++-12** (gcc-13 is also present; norm build uses g++-12).
- GPU: **NVIDIA RTX 6000 Ada (sm_89)**. No Blackwell, so **sm_100a is build-only**
  here (compile check + build-time delta, no kernel execution).
- The original `.venv` (python 3.12) has broken shebangs (`/home/cgomes/...`); only
  python 3.13 (via uv) is usable. The common library builds fine with
  `-DPython_EXECUTABLE=/home/vscode/.local/bin/python3` (python is only used for
  codegen scripts; pybind11 is not needed for the common lib).
- `nccl.h` is only in the pip `nvidia-nccl` wheel here. `logging.h` includes it
  unconditionally but CMake only adds the NCCL include dir under cuBLASMp/cuSolverMp
  (both OFF). Worked around for the build by putting the wheel's include dir on
  `CPATH`. (Pre-existing packaging gap, unrelated to this MR.)

Reconfigure used:

```bash
NCCL=/workspace/.venv/lib/python3.12/site-packages/nvidia/nccl
ln -sf libnccl.so.2 $NCCL/lib/libnccl.so
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake -S transformer_engine/common -B build/cmake \
  -DPython_EXECUTABLE=/home/vscode/.local/bin/python3 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DNVTE_BUILD_LEGACY_STATIC_NORM=OFF \
  -DCMAKE_PREFIX_PATH="$NCCL" -GNinja
export CPATH=$NCCL/include:$CPATH    # needed for the build + test compiles
```

### Build result (sm_89, norm RTC = NVTE_BUILD_LEGACY_STATIC_NORM=OFF)

Clean build of `libtransformer_engine.so`: **wall ~101 s**, `.so` shrank
**72.3 MB → 64.7 MB** (static norm fanout removed). Incremental rebuilds after a
header change that only touches the RTC string headers are ~33 s (a header that is
widely included, e.g. `utils.cuh`, triggers a fuller ~100 s rebuild).

### RTC norm correctness bugs found and fixed (all in this branch's in-progress work)

The norm RTC path had **never been executed** before (the prior container had no
GPU). Running the C++ `test_normalization` suite on the Ada GPU surfaced three RTC
compile bugs, since the RTC kernel sources are compiled by NVRTC at first use and
do **not** include `common.h` (it pulls in cuDNN/cutlass host headers):

1. **Missing dtype aliases.** The generated name expressions reference
   `::transformer_engine::normalization::{fp16,bf16,fp32,fp8e4m3,fp8e5m2}`, which
   are defined only in `common.h`. Added them under `#ifdef __CUDACC_RTC__` in
   `normalization/kernel_traits.h` (the header all four RTC sources include first).
2. **`std::is_same` / `std::conditional_t` unavailable.** `utils.cuh` deliberately
   avoids `<type_traits>` under NVRTC. Added a minimal NVRTC-only definition of
   `std::is_same` and `std::conditional[_t]` in the `__CUDACC_RTC__` branch of
   `utils.cuh`, so the shared kernel headers compile unchanged.
3. **Zero-length array rejected by NVRTC.** `rmsnorm_bwd_kernels.cuh`'s `dx_add_t`
   union uses `char _padding[sizeof(dx_t)-sizeof(add_t)]`, which is 0 when
   `sizeof(dx_t)==sizeof(add_t)` (e.g. fp16->fp16 fused backward+add). nvcc allows
   zero-length arrays (GNU ext); NVRTC does not. Added a `NeedsPadding=false`
   partial specialization with no padding member, keeping the static-build layout
   identical.

Also fixed an unrelated build break in `tests/cpp/operator/test_softmax.cu`: an
uncommitted `#include "common.h"` made the test's `Tensor` ambiguous with
`transformer_engine::Tensor`. Replaced the one symbol it needed
(`TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT`) with a file-local 16-bit dispatch macro
using the test harness's own `fp16`/`bf16` aliases, and dropped the include.

After (1)+(2) the full RMSNorm forward path compiles and passes; after (3) the
fused backward+add path compiles. Norm suite then reaches **136 OK / 0 FAILED**
(rest skipped by the test matrix) before hitting the issue below.

### Open issue: non-deterministic segfault in long norm runs

Running the full `*Norm*` operator suite segfaults (rc=139) after ~133–136 passing
cases. The crash point is **non-deterministic** (different test each run; not always
a fused-add case) and does **not** reproduce in a bf16-only subset or when the
crashing case is run in isolation — it needs the full fp16+bf16+fp32 mix
(more compiled kernel variants). 0 correctness failures before the crash. Working
hypothesis: memory corruption (likely OOB from mis-sized barrier/workspace in a
multi-CTA cooperative path) accumulating across kernel variants. Under cuda-gdb
(serialized) it runs further, consistent with a memory/timing-sensitive fault.
Investigation with cuda-gdb / compute-sanitizer in progress.

### Resolution of the segfault: NVRTC-internal, not our code or the machine

Classified the crash with several tools (all on the RTX 6000 Ada, sm_89):

- **Not the machine.** 125 GB RAM (117 GB free), 48 GB GPU memory (48 GB free),
  32 cores. The process crashes at only ~1.3 GB RSS — nowhere near any limit.
- **Kernels are correct.** Rebuilt with `NVTE_BUILD_LEGACY_STATIC_NORM=ON` (static,
  no NVRTC) and ran the full `*Norm*` suite: **192 OK / 0 FAILED**. So the math and
  the test matrix are fine; the RTC kernels themselves also produce correct results
  (every case that runs before the crash passes; `compute-sanitizer memcheck`
  reports **0 device memory errors**).
- **No host heap corruption detected.** Re-ran the RTC build under
  `GLIBC_TUNABLES=glibc.malloc.check=3 MALLOC_CHECK_=3`: still crashes, but glibc
  reports **no** invalid-free/corruption diagnostics.
- **The fault is inside NVRTC.** Core-dump backtrace:
  `nvrtcCompileProgram` (libnvrtc.so.12, frames #0–#11, no symbols) ←
  `KernelManager::compile` (#12) ← `register_launcher<ForwardKernelParams>` lambda
  (#13) ← `layernorm_fwd` (#15). Our call into NVRTC is a normal
  `nvrtcCompileProgram`; the fault is in NVRTC's own compiler internals.
- **It is cumulative, not per-kernel.** `*TeLayerNorm*` alone = 64 OK (clean),
  `*TeRmsNorm*` alone = 128 OK (clean), but the combined `*Norm*` (192 distinct
  kernels compiled in one process) crashes after ~130–190 compiles, at a
  non-deterministic point. Under cuda-gdb (serialized) the full 192 run to
  completion with no crash.

**Conclusion:** this is an intermittent fault in CUDA 12.8's NVRTC that only
manifests after compiling ~130+ distinct kernels in a single process — i.e., only
the exhaustive C++ test sweep hits it. Realistic workloads compile a handful of
norm shapes and will not. It is not caused by this branch's kernel code or by the
three RTC compile fixes above (those only made the kernels compile in the first
place). Recommended follow-ups (out of scope for this stage, and already
anticipated by the plan's "defer persistent cache" note): a persistent on-disk
cubin cache so repeat processes don't recompile, and/or `extern "C"` named kernel
wrappers to avoid `nvrtcAddNameExpression`/`nvrtcGetLoweredName` (the heaviest
NVRTC path). Per maintainer guidance, not investigating further now.

### sm_89 legacy-vs-RTC results on this container (CUDA 12.8)

| Mode | Build (norm TUs + link, incremental) | `libtransformer_engine.so` | Full `*Norm*` suite |
| --- | --: | --: | --- |
| Legacy static (`NVTE_BUILD_LEGACY_STATIC_NORM=ON`) | 107 s | 82.4 MB | 192 OK / 0 FAIL |
| NVRTC (`OFF`) | 103 s | 64.7 MB | 192 OK serialized; intermittent NVRTC crash in bulk (see above) |

The `.so` shrinks **82.4 MB → 64.7 MB (-21%)** from removing the static norm
template fanout. Wall-clock delta on sm_89 is small, exactly as the critical-path
analysis predicted (norm is not the sm_89 long pole; `cast_transpose_fusion.cu` /
activations are).

### sm_100a build check (compile-only; no Blackwell GPU here)

Built the common library for `-DCMAKE_CUDA_ARCHITECTURES=100a` (CUDA 12.8, g++-12)
in a separate `build/cmake100` tree to confirm the norm RTC sources compile for
Blackwell and to record the build-time delta the maintainer asked for.

Two notable results:

1. **The full-library sm_100a build is blocked by an *unrelated* TU:**
   `activation/gelu.cu` makes nvcc's `cicc` **segfault** ("Segmentation fault
   (core dumped)") at sm_100a. This is exactly the activation kernel the updated
   plan now prioritizes for NVRTC — at sm_100a its template/fast-path fanout is
   heavy enough to crash the compiler. The norm RTC TUs all compiled successfully
   *before* gelu in the same build.

2. **Norm 4-TU A/B at sm_100a (isolated, same 4 TUs):**

   | Mode | Wall (4 TUs) | Objects | ln_fwd_cuda_kernel.cu |
   | --- | --: | --: | --- |
   | NVRTC (`OFF`) | **32 s** | **0.47 MB total** (115–128 KB each) | compiles fine (122 KB) |
   | Legacy static (`ON`) | n/a | 11.8 MB for the 3 that built (2.9–6.3 MB each) | **cicc Segmentation fault — does not compile** |

   At sm_100a the legacy static `ln_fwd_cuda_kernel.cu` **crashes the compiler**
   (same cicc segfault as gelu), so the static path is not even buildable for that
   kernel on this toolchain, whereas the RTC path builds all four norm kernels into
   ~0.47 MB total. The remaining three legacy norm TUs are 2.9–6.3 MB objects vs
   115–128 KB under RTC (~25–50× smaller). So for sm_100a the norm RTC migration is
   not just a speed-up — it removes a hard compiler-crash blocker for the norm
   forward kernel.

(The intermittent NVRTC-compile-volume crash from the previous section is an sm_89
runtime-compilation observation; the sm_100a notes above are pure build-time and
done on the sm_89 host with no kernel execution.)
