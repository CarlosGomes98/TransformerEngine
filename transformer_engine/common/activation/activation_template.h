/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_template.h
 *  \brief Activation functions template.
 */

#ifndef TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
#define TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_

#include <cuda_runtime.h>
#include <transformer_engine/activation.h>

#include "../cast/dispatch/gated.cuh"
#include "../cast/dispatch/quantize.cuh"
#include "../common.h"
#include "./rtc_dispatch.h"

// When 1 (default), activations are dispatched through the statically-instantiated
// quantize templates. When 0 (CMake: NVTE_BUILD_LEGACY_STATIC_ACTIVATION=OFF), the
// non-gated forward/backward elementwise activations are compiled on demand via
// NVRTC, removing their (dtype x scaling-mode) template fanout from the build.
#ifndef NVTE_BUILD_LEGACY_STATIC_ACTIVATION
#define NVTE_BUILD_LEGACY_STATIC_ACTIVATION 1
#endif

namespace transformer_engine {

template <NVTE_Activation_Type ACT, typename ComputeType, typename Param,
          ComputeType (*OP)(ComputeType, const Param &)>
void act_fn(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
#if NVTE_BUILD_LEGACY_STATIC_ACTIVATION
  using namespace detail;
  constexpr bool IS_ACT = true;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, OP>(input, output, nullptr, stream);
#else
  rtc_act::run(input, nullptr, output, rtc_act::fwd_act_id(ACT), /*is_bwd=*/false, stream);
#endif
}

template <NVTE_Activation_Type ACT, typename ComputeType, typename Param,
          ComputeType (*OP)(ComputeType, const Param &)>
void dact_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output,
             cudaStream_t stream) {
#if NVTE_BUILD_LEGACY_STATIC_ACTIVATION
  using namespace detail;
  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = true;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  dispatch::quantize_bwd_helper<IS_DBIAS, IS_DACT, Empty, OP>(grad, input, output, dbias, workspace,
                                                              nullptr, stream);
#else
  rtc_act::run(input, grad, output, rtc_act::bwd_act_id(ACT), /*is_bwd=*/true, stream);
#endif
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &)>
void gated_act_fn(const NVTETensor input, NVTETensor output, Param &p, cudaStream_t stream) {
  using namespace detail;
  dispatch::quantize_gated_fwd_helper<Param, ActOP>(input, output, p, stream);
}

template <typename ComputeType, typename Param, ComputeType (*ActOP)(ComputeType, const Param &),
          ComputeType (*DActOP)(ComputeType, const Param &)>
void dgated_act_fn(const NVTETensor grad, const NVTETensor input, NVTETensor output, Param &p,
                   cudaStream_t stream) {
  using namespace detail;
  dispatch::quantize_gated_bwd_helper<Param, ActOP, DActOP>(grad, input, output, p, stream);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_ACTIVATION_TEMPLATE_H_
