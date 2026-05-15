# NVRTC Kernel Compilation Plan

## Goal

Reduce Transformer Engine build time by moving selected template-heavy CUDA kernels from
build-time instantiation to runtime specialization with NVRTC. The existing transpose RTC
path proves the approach is viable, so this plan extends that model to other kernels with
large template products while preserving static fallbacks until each migration is measured
and stable.

The implementation should be split into MR-sized phases. The order below favors the best
combination of build-time impact, implementation simplicity, and runtime risk. The largest
and simplest wins come first; harder or header-heavy migrations are later even when their
raw build-time impact is high.

## Current State

TE already has an NVRTC path in `transformer_engine/common/util/rtc.{h,cpp}` with:

- Per-device kernel cache in `KernelManager`.
- Compile support for generated source-string headers.
- Built-in include content for `utils.cuh` and `util/math.h`.
- Environment switch `NVTE_DISABLE_NVRTC`.
- Existing users under `transformer_engine/common/transpose/rtc/`.

Existing RTC-enabled transpose launchers are a good template for rollout:

- `transpose/transpose.cu`
- `transpose/cast_transpose.cu`
- `transpose/cast_transpose_fusion.cu`
- `transpose/swap_first_dims.cu`

## Affected Kernels, Sorted by Estimated Build-Time Impact

Impact estimates are based on source size, explicit registration count, template dispatch
fanout, CMake comments about compile-time hotspots, and local inspection of template axes.
They should be validated with `.ninja_log` or CMake timing before and after each MR.

| Rank | Kernel area | Estimated build-time impact | Migration complexity | Notes |
| --- | --- | --- | --- | --- |
| 1 | Normalization forward/backward (`normalization/*`) | Very high | Medium | Hundreds of `REGISTER_NORM_LAUNCHER` instantiations across LayerNorm and RMSNorm forward/backward. Good target because host planning and launch policy are already centralized. |
| 2 | Activation plus quantize dispatch (`activation/*`, `cast/dispatch/*`, `cast/mxfp8/*`, `cast/nvfp4/*`) | Very high | High | Small activation TUs trigger broad FP8/MXFP8/NVFP4 template products. Highest payoff after normalization, but more dtype/scaling/layout axes. |
| 3 | Hadamard transform kernels (`hadamard_transform/*`) | High | Very high | Large TUs and arch-specific compilation. Likely strong payoff, but CUTLASS/CuTe-style include dependencies make NVRTC packaging harder. |
| 4 | Fused softmax (`fused_softmax/*`) | High | Low to medium | Many shape/mask/log2 template cases, relatively self-contained kernels. Best early proof beyond transpose. |
| 5 | Blockwise and NVFP4 quantize-transpose (`transpose/quantize_transpose_*`, `cast/nvfp4/*`) | Medium-high | Medium-high | Arch-specific files with CUDA-version-dependent paths. Good after quantize RTC conventions are in place. |
| 6 | Swizzle kernels (`swizzle/swizzle.cu`) | Medium | Medium | Large file with repeated dtype/layout/tile cases. Runtime compile should be straightforward if source can be isolated cleanly. |
| 7 | Multi-tensor optimizer kernels (`multi_tensor/adam.cu` and related) | Medium | Medium | Nested dtype/optimizer mode dispatch. Useful, but less central to common TE import/build pain than norm/quantize paths. |
| 8 | Existing transpose RTC cleanup (`transpose/rtc/*` users) | Low direct impact | Low | Already moved to RTC. Use as the reference implementation; clean up only when a follow-on migration exposes a concrete need. |

## Phase 0: Baseline and RTC Ground Rules

MR scope:

- Capture a build-time baseline using `.ninja_log`, CMake timing, or another repeatable
  local build report.
- Add a small reporting script or documented command that ranks CUDA TUs by elapsed compile
  time.
- Use the checked-in `.ninja_log` helper for repeatable local reporting:
  `python3 build_tools/compile_time_report.py --ninja-log <build>/.ninja_log --filter
  'fused_softmax|util/rtc' --markdown`.
- Document the current RTC extension points:
  - generated source-string headers
  - `KernelManager` compile/cache behavior
  - `NVTE_DISABLE_NVRTC` fallback behavior
  - existing transpose RTC launch pattern
- Define lightweight conventions for follow-on MRs:
  - what belongs in an RTC compile key
  - how new RTC source-string headers should be generated
  - how fallback paths should be tested
  - how first-call compile latency should be reported
- Avoid implementing new `KernelManager` features in Phase 0 unless the baseline work
  exposes a correctness issue in existing RTC users.
- Document how to add a new RTC source file using the transpose implementation as the
  canonical example.

Acceptance criteria:

- Existing transpose RTC tests still pass.
- `tests/cpp/util/test_nvrtc.cpp` still covers the current supported RTC behavior.
- Existing transpose, cast-transpose, fused cast-transpose, and swap-first-dims tests pass.
- Static build output is unchanged for kernels not yet migrated.
- The baseline report identifies the current top compile-time TUs.

KernelManager extension policy:

- Add `KernelManager` capabilities only in the first MR that needs them.
- Keep each extension as small as possible and covered by a focused RTC utility test.
- Defer persistent disk cache until a migrated kernel shows first-call compile latency or
  repeated-process cache misses that justify it.
- Defer broad cache-key restructuring until a migrated kernel needs compile options or
  source/header combinations that the current key cannot represent safely.
- Defer CUDA graph capture changes until a migrated path is actually reached during capture
  or needs an explicit guard.

## Phase 1: Fused Softmax RTC

Why first:

Fused softmax has a strong impact-to-complexity ratio. The kernels are shape-specialized,
mostly self-contained, and do not require the heavy quantization or normalization planning
machinery. This is the best early non-transpose proof point.

MR scope:

- Add RTC source files for:
  - `scaled_softmax`
  - `scaled_masked_softmax`
  - `scaled_upper_triang_masked_softmax`
  - `scaled_aligned_causal_masked_softmax`
- Replace compile-time log2/shape/mask template fanout with runtime compilation keyed by:
  - dtype
  - batch/head dimensions as needed by the launch policy
  - `log2_elements`
  - mask/causal/alignment mode
  - forward/backward mode
- Keep host-side validation, shape checks, and launch-policy selection in C++.
- Preserve static fallback behind `NVTE_DISABLE_NVRTC` and, initially, a build-time option
  such as `NVTE_BUILD_LEGACY_STATIC_KERNELS`.

Acceptance criteria:

- C++ fused softmax tests pass with RTC enabled and disabled.
- JAX softmax custom-call tests pass where applicable.
- Build timing shows reduced compile time for `common/fused_softmax/*`.
- First-call compile latency is logged and acceptable for representative shapes.

## Phase 2: Normalization RTC Registry

Why next:

Normalization appears to be the largest static-specialization hotspot. It has hundreds of
registered launchers, but the registration/plan pattern gives us a natural place to replace
function-pointer tables with runtime-compiled kernels.

MR scope:

- Convert LayerNorm and RMSNorm launcher registration from concrete kernel function
  pointers to descriptors that can produce an RTC compile key.
- Add RTC sources for the major forward/backward paths:
  - LayerNorm forward
  - LayerNorm backward semi/atomic paths
  - RMSNorm forward
  - RMSNorm backward semi/atomic paths
- Keep host planning, workspace sizing, persistent-kernel eligibility, and dispatch
  decisions in normal C++.
- Compile only the selected descriptor for a runtime shape/dtype combination instead of
  instantiating every registered case at build time.
- Keep static registered kernels as fallback for one release cycle or until the measured
  runtime behavior is stable.

Likely compile key fields:

- norm type: LayerNorm or RMSNorm
- direction: forward or backward
- input/output dtype and parameter dtype
- hidden size bucket or exact hidden size, depending on current tuning tables
- rows-per-CTA / columns-per-CTA / warp count
- zero-centered gamma and other boolean options
- architecture and CUDA version

Acceptance criteria:

- Existing C++ normalization tests pass with RTC enabled and disabled.
- PyTorch and JAX normalization coverage passes for representative dtypes and hidden sizes.
- Build timing shows a clear reduction for `ln_*` and `rmsnorm_*` CUDA TUs.
- Runtime first-call latency and cache hit behavior are visible in debug logs.

## Phase 3: Activation and FP8/MXFP8 Quantize RTC

Why after normalization:

This area has very high build-time impact, but the dispatch space is wider. It should use
the proven softmax/norm RTC conventions before removing large static template products.

MR scope:

- Migrate activation-driven quantize kernels that currently fan out through
  `cast/dispatch/*`.
- Use enum-based activation selection in RTC source, following the existing
  `cast_transpose_fusion` RTC approach instead of template function pointers.
- Start with delayed-scaling FP8 cast/activation paths, then extend to MXFP8:
  - plain activation forward/backward
  - gated activation forward/backward
  - bias and dbias variants
  - rowwise, columnwise, and bidirectional scaling modes
- Keep the most specialized/tuned static MXFP8 kernels until generic RTC coverage is
  measured.

Likely compile key fields:

- input dtype, output dtype, scale dtype
- quantization mode: FP8 delayed scaling, MXFP8, or generic block scaling
- activation type and direction
- bias/dbias/dactivation flags
- rowwise/columnwise/bidirectional output
- layout and transpose flags
- block/chunk dimensions
- architecture and CUDA version

Acceptance criteria:

- `tests/cpp/operator/test_act.cu`
- `tests/cpp/operator/test_cast_dbias_dgelu.cu`
- `tests/cpp/operator/test_cast_gated_swiglu.cu`
- `tests/cpp/operator/test_cast_mxfp8.cu`
- `tests/cpp/operator/test_cast_mxfp8_grouped.cu`
- Representative PyTorch/JAX activation and quantize tests.
- Build timing shows reduction in `activation/*` and cast/quantize-heavy TUs.

## Phase 4: NVFP4 and Blockwise Quantize-Transpose RTC

Why after Phase 4:

NVFP4/blockwise quantize shares many dispatch concepts with activation/quantize but adds
more architecture and CUDA-version constraints. It is easier once compile-key conventions
for quantization modes are already established.

MR scope:

- Add RTC coverage for:
  - `quantize_transpose_vector_blockwise_fp4`
  - `quantize_transpose_square_blockwise`
  - NVFP4 quantize-transpose paths
  - grouped NVFP4 quantize-transpose paths
- Keep CUDA-version feature checks host-side and include the selected capability in the
  compile key.
- Preserve tuned static kernels for edge cases until RTC coverage is complete.

Acceptance criteria:

- NVFP4 and block-scaling C++ tests pass with RTC enabled and disabled.
- CUDA-version-gated paths fail cleanly or fall back when unsupported.
- Build timing shows reduced compile time for the blockwise transpose and NVFP4 TUs.

## Phase 5: Swizzle RTC

Why here:

Swizzle has meaningful compile-time weight and a contained launch surface, but it is not as
large a build-time multiplier as normalization or quantize dispatch.

MR scope:

- Isolate swizzle kernel device code into RTC source-string files.
- Compile by dtype, swizzle mode, tile geometry, and layout.
- Keep host validation and layout selection static.
- Preserve static fallback for unsupported or low-frequency modes.

Acceptance criteria:

- Existing swizzle tests pass with RTC enabled and disabled.
- Build timing shows a reduction for `common/swizzle/swizzle.cu`.
- First-call latency is acceptable for common swizzle modes.

## Phase 6: Hadamard Transform RTC

Why later:

Hadamard kernels may be high-impact, but they are more likely to require careful handling of
large template headers and architecture-specific tuning. They should wait until the RTC
cache, fallback, and debugging story is mature.

MR scope:

- Start with the simplest row/column Hadamard transform kernels.
- Evaluate whether the necessary device code can be packaged into generated source-string
  headers without pulling in excessive host-only dependencies.
- Add RTC variants for common dtype/layout/scale combinations.
- Keep CUTLASS/CuTe-heavy or rarely used specializations static until measured.

Acceptance criteria:

- Existing Hadamard transform tests pass with RTC enabled and disabled.
- Runtime compile time is not excessive, with persistent cache considered only if this phase
  shows that it is needed.
- Build timing shows a measurable reduction for Hadamard transform TUs.

## Phase 7: Multi-Tensor and Remaining Template Hotspots

Why last:

These kernels are useful cleanup targets, but they are lower priority than the core
normalization, softmax, and quantization paths.

MR scope:

- Inspect build-time reports after Phases 1-7.
- Select remaining hotspots by measured compile-time cost, not by source size alone.
- Candidates include:
  - `multi_tensor/adam.cu`
  - remaining cast/dequantize variants
  - any framework-specific custom-call glue that still instantiates large template sets

Acceptance criteria:

- Each selected hotspot has before/after build-time evidence.
- No static fallback is removed until RTC and fallback paths have equivalent test coverage.

## Cross-Cutting Design Rules

- Keep host-side shape validation, error handling, workspace planning, and launch-policy
  selection in compiled C++.
- Move only the final specialized device kernel body and launch wrapper into RTC source
  where possible.
- Use explicit enum and integer template axes in RTC source instead of C++ function pointer
  template parameters.
- Make every new or changed RTC compile key auditable in debug logs.
- Cache new migrated paths by enough source, header, and compile-option identity to prevent
  stale kernel reuse.
- Keep `NVTE_DISABLE_NVRTC` working for every migrated path.
- Prefer a static fallback until the migrated path has tests, timing data, and at least one
  release cycle of confidence.
- Add new generated RTC source headers through CMake, following the existing transpose
  convention.

## Suggested MR Sequence

1. Baseline report plus lightweight RTC conventions.
2. Fused softmax RTC.
3. Normalization RTC registry.
4. Activation plus FP8/MXFP8 quantize RTC, first covering common delayed-scaling and MXFP8
   paths.
5. NVFP4 and blockwise quantize-transpose RTC.
6. Swizzle RTC.
7. Hadamard transform RTC feasibility and first common kernels.
8. Multi-tensor and remaining measured hotspots.

## Validation Matrix

For each migration MR:

- Build with RTC enabled.
- Build with `NVTE_DISABLE_NVRTC=1` or the static fallback option enabled.
- Run targeted C++ operator tests for the migrated kernel family.
- Run at least one PyTorch and/or JAX integration test that reaches the migrated path.
- Measure before/after build time for affected TUs.
- Record first-call compile latency and cache-hit latency for representative shapes.
- Check that CUDA graph capture behavior is unchanged or explicitly guarded.

## Open Questions

- Should persistent RTC cache be enabled by default, or only when a cache directory env var
  is set?
- How long should static fallback kernels remain in the build after an RTC path is proven?
- Which build configuration should be the official compile-time benchmark:
  PyTorch-only, JAX-only, or both frameworks?
- Do we want a CI job that exercises `NVTE_DISABLE_NVRTC=1`, or is local/nightly coverage
  sufficient?
- Should runtime compilation failures always fall back to static kernels when available, or
  should debug builds fail fast to expose RTC source regressions?
