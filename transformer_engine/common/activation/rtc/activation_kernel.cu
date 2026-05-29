/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC source for elementwise (non-gated) activation forward/backward with
// optional FP8 per-tensor (delayed-scaling) rowwise quantization.
//
// This file is *not* compiled by nvcc at build time. It is embedded as a string
// header (see CMakeLists.txt make_string_header_from_file) and compiled at
// runtime by NVRTC for the exact (activation, fwd/bwd, in-dtype, out-dtype)
// combination requested. It therefore must only include NVRTC-safe headers:
// utils.cuh (CUDA fp16/bf16/fp8 types + helpers) and util/math.h (the device
// activation ops). It must NOT include common.h (pulls in cuDNN/CUTLASS host
// headers).

#include "utils.cuh"
#include "util/math.h"

namespace transformer_engine {
namespace rtc_act {

// dtype aliases referenced by the host-built name expressions
// (e.g. ::transformer_engine::rtc_act::fp16). Underlying CUDA types come from
// utils.cuh.
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

// Activation identifiers. The host dispatcher passes the matching integer as a
// non-type template argument in the kernel name expression, so the ordering
// here is part of the ABI between rtc_dispatch.cpp and this source.
enum ActId {
  ACT_GELU = 0,
  ACT_DGELU = 1,
  ACT_QGELU = 2,
  ACT_DQGELU = 3,
  ACT_SILU = 4,
  ACT_DSILU = 5,
  ACT_RELU = 6,
  ACT_DRELU = 7,
  ACT_SRELU = 8,
  ACT_DSRELU = 9,
  ACT_SIGMOID = 10,
  ACT_DSIGMOID = 11
};

template <int ACT>
__device__ inline float apply_act(float v) {
  Empty e{};
  if (ACT == ACT_GELU) return gelu<float, float>(v, e);
  if (ACT == ACT_DGELU) return dgelu<float, float>(v, e);
  if (ACT == ACT_QGELU) return qgelu<float, float>(v, e);
  if (ACT == ACT_DQGELU) return dqgelu<float, float>(v, e);
  if (ACT == ACT_SILU) return silu<float, float>(v, e);
  if (ACT == ACT_DSILU) return dsilu<float, float>(v, e);
  if (ACT == ACT_RELU) return relu<float, float>(v, e);
  if (ACT == ACT_DRELU) return drelu<float, float>(v, e);
  if (ACT == ACT_SRELU) return srelu<float, float>(v, e);
  if (ACT == ACT_DSRELU) return dsrelu<float, float>(v, e);
  if (ACT == ACT_SIGMOID) return sigmoid<float, float>(v, e);
  if (ACT == ACT_DSIGMOID) return dsigmoid<float, float>(v, e);
  return 0.0f;
}

// Atomic max for non-negative IEEE-754 floats (amax is always >= 0, so the
// bit pattern ordering matches the float ordering).
__device__ inline void atomic_amax(float *addr, float val) {
  atomicMax(reinterpret_cast<int *>(addr), __float_as_int(val));
}

// Scalar grid-stride elementwise activation kernel. Compute is always in fp32.
// For backward (IS_BWD) the activation derivative is multiplied by the incoming
// gradient. For FP8 output, `scale` (in) is applied and `scale_inv`/`amax`
// (out) are produced; for high-precision output those pointers are null.
template <int ACT, bool IS_BWD, typename IType, typename OType>
__global__ void act_kernel(const IType *__restrict__ x, const IType *__restrict__ grad,
                           OType *__restrict__ y, const float *scale, float *amax, float *scale_inv,
                           const unsigned long long N) {  // NOLINT(runtime/int)
  const float s = (scale != nullptr) ? *scale : 1.0f;
  const bool need_amax = (amax != nullptr);
  float local_amax = 0.0f;
  const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
    float out = apply_act<ACT>(static_cast<float>(x[i]));
    if (IS_BWD) {
      out *= static_cast<float>(grad[i]);
    }
    if (need_amax) {
      local_amax = fmaxf(local_amax, fabsf(out));
    }
    if (scale != nullptr) {
      out *= s;
    }
    y[i] = static_cast<OType>(out);
  }
  if (need_amax) {
    atomic_amax(amax, local_amax);
  }
  if (scale_inv != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    *scale_inv = 1.0f / s;
  }
}

}  // namespace rtc_act
}  // namespace transformer_engine
