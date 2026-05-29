/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_ACTIVATION_RTC_DISPATCH_H_
#define TRANSFORMER_ENGINE_COMMON_ACTIVATION_RTC_DISPATCH_H_

#include <cuda_runtime.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/transformer_engine.h>

namespace transformer_engine {
namespace rtc_act {

// RTC activation identifiers. Must match the ActId enum in
// activation/rtc/activation_kernel.cu (the value is emitted into the NVRTC
// kernel name expression).
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

// Forward / backward (derivative) RTC activation id for a given activation type.
int fwd_act_id(NVTE_Activation_Type act);
int bwd_act_id(NVTE_Activation_Type act);

// Run a non-gated elementwise activation through NVRTC.
//   forward:  out = act(x)            (pass grad = nullptr, is_bwd = false)
//   backward: out = act'(x) * grad    (pass grad,           is_bwd = true)
// For FP8 (delayed/per-tensor) output the scale/amax/scale_inv metadata on
// `output` is consumed/produced; high-precision output ignores it.
void run(const NVTETensor x, const NVTETensor grad, NVTETensor output, int act_id, bool is_bwd,
         cudaStream_t stream);

}  // namespace rtc_act
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_ACTIVATION_RTC_DISPATCH_H_
