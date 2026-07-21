/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rtc_dispatch.cuh
 *  \brief Host-side NVRTC dispatch for the specialized MXFP8 cast-only kernels.
 */

#ifndef TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_
#define TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_

#if !defined(__CUDACC_RTC__)

#include <cuda.h>

#include <string>

#include "../../../util/rtc.h"
// NB: do not include util/string.h here — it pulls in <regex>, which is heavy
// enough to ICE the device compiler (cicc) when this header is parsed inside a
// large .cu TU. Label building uses plain std::string; regex_replace lives in
// the host-only rtc_dispatch.cpp.
#include "quantize_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

// Compile (if not already cached) the rowwise cast-only RTC kernel for the given
// element-type spellings. Defined in rtc_dispatch.cpp, a host-only translation
// unit, so the large embedded kernel sources are never parsed by the device
// compiler (cicc) of TUs that merely launch the kernel.
void compile_rowwise_cast_only_rtc(const std::string &kernel_label, const std::string &itype_name,
                                   const std::string &otype_name);
void compile_bidimensional_cast_only_rtc(const std::string &kernel_label,
                                         const std::string &itype_name,
                                         const std::string &otype_name, int num_stages, int iter_n,
                                         bool use_cvt_4x);

// Element-type spellings used both for the __ITYPE__/__OTYPE__ substitution and
// as part of the compiled-kernel cache key. The names must resolve inside the
// NVRTC translation unit (see ptx.cuh / utils.cuh RTC aliases). The generic
// fallback (via detail::type_name) covers types the switch instantiates but the
// rowwise path never actually launches (e.g. float).
template <typename T>
inline const char *rtc_type_name() {
  return detail::type_name<T>();
}
template <>
inline const char *rtc_type_name<fp16>() {
  return "fp16";
}
template <>
inline const char *rtc_type_name<bf16>() {
  return "bf16";
}
template <>
inline const char *rtc_type_name<fp8e4m3>() {
  return "fp8e4m3";
}
template <>
inline const char *rtc_type_name<fp8e5m2>() {
  return "fp8e5m2";
}

// Compile (on first use) and launch the 1x32 rowwise cast-only kernel via NVRTC.
// Geometry is derived from CastTraits here (host code, no device sources), while
// the actual NVRTC compilation lives in the host-only .cpp.
template <typename IType, typename OType>
inline void launch_rowwise_cast_only_rtc(IType *input, OType *output, e8m0_t *scales_rowwise,
                                         const float *noop, int32_t rows, int32_t cols,
                                         int32_t scale_stride_rowwise, int32_t scale_stride_colwise,
                                         cudaStream_t stream) {
  using traits = CastTraits<IType, OType, /*rowwise=*/true, /*colwise=*/false>;

  const std::string itype_name = rtc_type_name<IType>();
  const std::string otype_name = rtc_type_name<OType>();

  // The cache key encodes everything that varies the compiled kernel. Bump the
  // tiling tag when the traits geometry becomes runtime-selectable (Phase 2).
  const std::string kernel_label = std::string("quantize_mxfp8_rowwise_cast_only,itype=") +
                                   itype_name + ",otype=" + otype_name + ",tiling=v1";

  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_rowwise_cast_only_rtc(kernel_label, itype_name, otype_name);
  }

  if (traits::smem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(traits::smem));
  }

  dim3 block(traits::threadLayout::num, traits::warpLayout::N, traits::warpLayout::M);
  dim3 grid((cols + traits::blockDimN - 1) / traits::blockDimN,
            (rows + traits::blockDimM - 1) / traits::blockDimM);
  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(traits::smem), stream, input,
             output, scales_rowwise, noop, rows, cols, scale_stride_rowwise,
             scale_stride_colwise);
}

// Per-shape compile-time config for the bidimensional kernel. Populate the table
// in bidim_config_for() from an offline autotune sweep
// (benchmarks/mxfp8_sweep_dsv3.sh). The default {2,4,true} reproduces the shipped
// tiling exactly, so unlisted shapes are unchanged.
struct BidimConfig {
  int32_t num_stages;
  int32_t iter_n;
  bool use_cvt_4x;
};

inline BidimConfig bidim_config_for(int32_t rows, int32_t cols) {
  // >>> Autotuned overrides go here, e.g.:
  //   if (rows == 4096 && cols == 512) return {3, 2, true};
  //   if (rows == 4096 && cols == 32768) return {2, 16, true};
  (void)rows;
  (void)cols;
  return {2, 4, true};  // shipped default (== CastTraits<...,true,true>)
}

// Compile+launch one concrete bidimensional config. Geometry/smem come straight
// from BidimTunableTraits, so every config is exactly what the JIT'd kernel uses.
template <typename IType, typename OType, int32_t NS, int32_t ITN, bool CVT>
inline void launch_bidim_impl(const CUtensorMap &tensor_map_input,
                              const CUtensorMap &tensor_map_rowwise_output,
                              const CUtensorMap &tensor_map_colwise_output, e8m0_t *scales_rowwise,
                              e8m0_t *scales_colwise, const float *noop, int32_t rows,
                              int32_t cols, int32_t scale_stride_rowwise,
                              int32_t scale_stride_colwise, cudaStream_t stream,
                              const std::string &itype_name, const std::string &otype_name) {
  using traits = BidimTunableTraits<IType, OType, NS, ITN, CVT>;
  const std::string kernel_label = std::string("quantize_mxfp8_bidimensional_cast_only,itype=") +
                                   itype_name + ",otype=" + otype_name + ",ns=" + std::to_string(NS) +
                                   ",itn=" + std::to_string(ITN) + ",cvt=" + (CVT ? "4x" : "2x");
  auto &mgr = rtc::KernelManager::instance();
  if (!mgr.is_compiled(kernel_label)) {
    compile_bidimensional_cast_only_rtc(kernel_label, itype_name, otype_name, NS, ITN, CVT);
  }
  if (traits::smem > 0) {
    mgr.set_function_attribute(kernel_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(traits::smem));
  }
  dim3 block(traits::rowThreadLayout::num, traits::numWarps);
  dim3 grid((cols + traits::blockDIM::N - 1) / traits::blockDIM::N,
            (rows + traits::blockDIM::M - 1) / traits::blockDIM::M);
  mgr.launch(kernel_label, grid, block, static_cast<unsigned int>(traits::smem), stream,
             tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,
             scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise);
}

// Compile (on first use) and launch the 32x32 bidimensional cast-only kernel via
// NVRTC, selecting the (numStages, iterN, cvt) config for this shape. TMA
// descriptors are built host-side by the caller (config-independent).
template <typename IType, typename OType>
inline void launch_bidimensional_cast_only_rtc(
    const CUtensorMap &tensor_map_input, const CUtensorMap &tensor_map_rowwise_output,
    const CUtensorMap &tensor_map_colwise_output, e8m0_t *scales_rowwise, e8m0_t *scales_colwise,
    const float *noop, int32_t rows, int32_t cols, int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise, cudaStream_t stream) {
  const std::string itype_name = rtc_type_name<IType>();
  const std::string otype_name = rtc_type_name<OType>();
  const BidimConfig c = bidim_config_for(rows, cols);

#define NVTE_MXFP8_BIDIM_CASE(NS, ITN, CVT)                                                      \
  if (c.num_stages == (NS) && c.iter_n == (ITN) && c.use_cvt_4x == (CVT)) {                      \
    launch_bidim_impl<IType, OType, NS, ITN, CVT>(                                               \
        tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,  \
        scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise, stream,    \
        itype_name, otype_name);                                                                 \
    return;                                                                                      \
  }
  // Enumerated configs (must match the autotuner grid). cvt fixed at 4x.
  NVTE_MXFP8_BIDIM_CASE(2, 1, true)
  NVTE_MXFP8_BIDIM_CASE(2, 2, true)
  NVTE_MXFP8_BIDIM_CASE(2, 4, true)
  NVTE_MXFP8_BIDIM_CASE(2, 8, true)
  NVTE_MXFP8_BIDIM_CASE(2, 16, true)
  NVTE_MXFP8_BIDIM_CASE(3, 1, true)
  NVTE_MXFP8_BIDIM_CASE(3, 2, true)
  NVTE_MXFP8_BIDIM_CASE(3, 4, true)
  NVTE_MXFP8_BIDIM_CASE(3, 8, true)
  NVTE_MXFP8_BIDIM_CASE(3, 16, true)
  NVTE_MXFP8_BIDIM_CASE(4, 1, true)
  NVTE_MXFP8_BIDIM_CASE(4, 2, true)
  NVTE_MXFP8_BIDIM_CASE(4, 4, true)
  NVTE_MXFP8_BIDIM_CASE(4, 8, true)
  NVTE_MXFP8_BIDIM_CASE(4, 16, true)
#undef NVTE_MXFP8_BIDIM_CASE

  // Unknown config -> shipped default.
  launch_bidim_impl<IType, OType, 2, 4, true>(
      tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,
      scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise, stream,
      itype_name, otype_name);
}

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // !__CUDACC_RTC__

#endif  // TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_
