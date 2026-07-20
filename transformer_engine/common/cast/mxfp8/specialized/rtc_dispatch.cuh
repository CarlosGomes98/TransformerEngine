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

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // !__CUDACC_RTC__

#endif  // TRANSFORMER_ENGINE_SPECIALIZED_MXFP8_RTC_DISPATCH_CUH_
