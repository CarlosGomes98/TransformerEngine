/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Runtime autotuner for the MXFP8 *bidimensional* (rowwise+colwise, TMA) cast
// kernel -- the one dsv3 exercises. For a given shape/dtype it JIT-compiles the
// kernel for a grid of compile-time tiling configs (pipeline stages x per-block
// iteration width x 4x/2x convert), benchmarks each on-device, checks they agree
// bit-for-bit, and prints a ranking. This is the payoff of the NVRTC migration:
// pick the best compile-time config per shape at runtime, then cache it.
//
// It reuses the library's create_2D_tensor_map (TMA descriptor build) and
// rtc::KernelManager (compile/launch), and derives geometry/smem directly from
// BidimTunableTraits, so nothing is re-implemented by hand.
//
// Build (from repo root, in the NGC container, against your built lib):
//   nvcc -std=c++17 -O3 --expt-relaxed-constexpr \
//        -I transformer_engine/common/.. -I transformer_engine/common/include \
//        -I <cudnn-frontend>/include \
//        -o mxfp8_autotune_bidim benchmarks/mxfp8_autotune_bidimensional.cu \
//        -L $REPO/build/te -ltransformer_engine -lnvrtc -lcuda
// Run:
//   TE_COMMON=$PWD/transformer_engine/common \
//   LD_LIBRARY_PATH=$REPO/build/te ./mxfp8_autotune_bidim 4096 7168 bf16 fp8e4m3

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/cast/mxfp8/specialized/quantize_mxfp8.cuh"
#include "common/common.h"
#include "common/util/rtc.h"
#include "common/util/string.h"

namespace te = transformer_engine;
namespace spec = transformer_engine::dispatch::mxfp8::quantize_kernel::specialized;

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e = (x);                                                                      \
    if (e != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

static std::string read_file(const std::string &p) {
  std::ifstream f(p);
  if (!f) {
    fprintf(stderr, "cannot open %s\n", p.c_str());
    exit(2);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Tunable bidimensional kernel source. extern "C" for a stable launch name.
static const char *kSrc = R"KSRC(
#include "specialized_quantize_mxfp8.cuh"
using namespace transformer_engine;
namespace spec = transformer_engine::dispatch::mxfp8::quantize_kernel::specialized;
using Traits = spec::BidimTunableTraits<__ITYPE__, __OTYPE__, __NS__, __ITN__, __CVT__>;
extern "C" __global__ void __launch_bounds__(Traits::numThreads) mxfp8_bidim_autotune_kernel(
    const __grid_constant__ CUtensorMap tmap_in, const __grid_constant__ CUtensorMap tmap_row,
    const __grid_constant__ CUtensorMap tmap_col, e8m0_t *scales_row, e8m0_t *scales_col,
    int rows, int cols, int ssr, int ssc) {
  spec::quantize_mxfp8_bidimensional_cast_only_body<Traits>(tmap_in, tmap_row, tmap_col,
                                                            scales_row, scales_col, rows, cols,
                                                            ssr, ssc);
}
)KSRC";

struct GlobalCtx {
  std::string itype_name, otype_name;
  int rows, cols, ssr, ssc, iters;
  CUtensorMap tmap_in, tmap_row, tmap_col;
  te::e8m0_t *d_scales_row, *d_scales_col;
  std::vector<te::rtc::Header> headers;  // extra in-memory headers for NVRTC
  std::vector<uint8_t> ref_row, ref_col;
  cudaStream_t stream;
};

struct Result {
  int ns, itn, cvt;
  double ms, gbps;
  bool correct, ok;
};

// One config: geometry from BidimTunableTraits, JIT-compile + launch + time.
template <int NS, int ITN, bool CVT>
static Result run_config(GlobalCtx &g) {
  // Geometry is independent of which 2-byte/1-byte dtype is used, so a
  // representative instantiation gives the correct block/grid/smem.
  using T = spec::BidimTunableTraits<te::bf16, te::fp8e4m3, NS, ITN, CVT>;
  Result r{NS, ITN, CVT, 0, 0, false, false};

  const std::string label = te::concat_strings("autotune_bidim,i=", g.itype_name, ",o=",
                                               g.otype_name, ",ns=", NS, ",itn=", ITN, ",cvt=", CVT);
  auto &mgr = te::rtc::KernelManager::instance();
  if (!mgr.is_compiled(label)) {
    std::string code = te::regex_replace(kSrc, "__ITYPE__", g.itype_name);
    code = te::regex_replace(code, "__OTYPE__", g.otype_name);
    code = te::regex_replace(code, "__NS__", std::to_string(NS));
    code = te::regex_replace(code, "__ITN__", std::to_string(ITN));
    code = te::regex_replace(code, "__CVT__", CVT ? "true" : "false");
    mgr.compile(label, "mxfp8_bidim_autotune_kernel", code, "autotune_bidim.cu",
                {"--device-int128", "-default-device"}, g.headers);
  }
  if (T::smem > 0) {
    mgr.set_function_attribute(label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               static_cast<int>(T::smem));
  }

  dim3 block(T::rowThreadLayout::num, T::numWarps);
  dim3 grid((g.cols + T::blockDIM::N - 1) / T::blockDIM::N,
            (g.rows + T::blockDIM::M - 1) / T::blockDIM::M);

  auto launch = [&]() {
    mgr.launch(label, grid, block, static_cast<unsigned>(T::smem), g.stream, g.tmap_in, g.tmap_row,
               g.tmap_col, g.d_scales_row, g.d_scales_col, g.rows, g.cols, g.ssr, g.ssc);
  };

  // warmup + correctness
  launch();
  if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
    fprintf(stderr, "[ns=%d itn=%d cvt=%d] launch failed: %s\n", NS, ITN, CVT,
            cudaGetErrorString(cudaGetLastError()));
    return r;
  }
  const size_t row_sz = (size_t)g.rows * g.ssr;
  const size_t col_sz = (size_t)((g.rows + 31) / 32) * g.cols;
  std::vector<uint8_t> cur_row(row_sz), cur_col(col_sz);
  CK(cudaMemcpy(cur_row.data(), g.d_scales_row, row_sz, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cur_col.data(), g.d_scales_col, col_sz, cudaMemcpyDeviceToHost));
  if (g.ref_row.empty()) {
    g.ref_row = cur_row;
    g.ref_col = cur_col;
    r.correct = true;
  } else {
    r.correct = (cur_row == g.ref_row) && (cur_col == g.ref_col);
  }

  // timing
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, g.stream));
  for (int i = 0; i < g.iters; i++) launch();
  CK(cudaEventRecord(e1, g.stream));
  CK(cudaEventSynchronize(e1));
  float total = 0;
  CK(cudaEventElapsedTime(&total, e0, e1));
  r.ms = total / g.iters;
  const double bytes = (double)g.rows * g.cols * 2 + (double)g.rows * g.cols * 2 + row_sz + col_sz;
  r.gbps = bytes / (r.ms * 1e-3) / 1e9;
  r.ok = true;
  printf("%-4d %-5d %-5s %10.4f %10.1f %8s\n", NS, ITN, CVT ? "4x" : "2x", r.ms, r.gbps,
         r.correct ? "yes" : "NO");
  return r;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <rows> <cols> <bf16|fp16> <fp8e4m3|fp8e5m2> [iters]\n", argv[0]);
    return 2;
  }
  GlobalCtx g;
  g.rows = atoi(argv[1]);
  g.cols = atoi(argv[2]);
  g.itype_name = argv[3];
  g.otype_name = argv[4];
  g.iters = argc > 5 ? atoi(argv[5]) : 50;
  g.ssr = (g.cols + 31) / 32;
  g.ssc = g.cols;

  const char *common = getenv("TE_COMMON");
  const std::string R = common ? std::string(common) + "/" : "transformer_engine/common/";
  // Header contents must outlive compile(); keep them in static storage.
  static std::string h0 = read_file(R + "cast/mxfp8/specialized/quantize_mxfp8.cuh");
  static std::string h1 = read_file(R + "util/ptx.cuh");
  static std::string h2 = read_file(R + "cast/mxfp8/specialized/state_counter.cuh");
  static std::string h3 = read_file(R + "cast/mxfp8/specialized/swizzle.cuh");
  g.headers = {{h0.c_str(), "specialized_quantize_mxfp8.cuh"},
               {h1.c_str(), "ptx.cuh"},
               {h2.c_str(), "state_counter.cuh"},
               {h3.c_str(), "swizzle.cuh"}};

  CK(cudaStreamCreate(&g.stream));

  const te::DType itype = g.itype_name == "fp16" ? te::DType::kFloat16 : te::DType::kBFloat16;
  const te::DType otype =
      g.otype_name == "fp8e5m2" ? te::DType::kFloat8E5M2 : te::DType::kFloat8E4M3;

  // Device buffers.
  void *d_in, *d_rowout, *d_colout;
  CK(cudaMalloc(&d_in, (size_t)g.rows * g.cols * 2));
  CK(cudaMalloc(&d_rowout, (size_t)g.rows * g.cols));
  CK(cudaMalloc(&d_colout, (size_t)g.rows * g.cols));
  CK(cudaMalloc(&g.d_scales_row, (size_t)g.rows * g.ssr));
  CK(cudaMalloc(&g.d_scales_col, (size_t)((g.rows + 31) / 32) * g.cols));
  {
    std::vector<uint16_t> h_in((size_t)g.rows * g.cols);
    for (size_t i = 0; i < h_in.size(); i++) h_in[i] = (uint16_t)((i * 2654435761u) >> 15);
    CK(cudaMemcpy(d_in, h_in.data(), h_in.size() * 2, cudaMemcpyHostToDevice));
  }

  // Build the 3 TMA descriptors once (box dims / swizzle are config-independent).
  using RefT = spec::BidimTunableTraits<te::bf16, te::fp8e4m3>;
  te::create_2D_tensor_map(g.tmap_in, te::SimpleTensor(d_in, {(size_t)g.rows, (size_t)g.cols}, itype),
                           g.rows, g.cols, RefT::blockIterDim::M, RefT::blockIterDim::N, g.cols, 0,
                           te::TypeInfo<te::bf16>::size, RefT::input_swizzle_pattern);
  te::create_2D_tensor_map(g.tmap_row,
                           te::SimpleTensor(d_rowout, {(size_t)g.rows, (size_t)g.cols}, otype),
                           g.rows, g.cols, RefT::blockIterDim::M, RefT::blockIterDim::N, g.cols, 0,
                           8, RefT::output_swizzle_pattern);
  te::create_2D_tensor_map(g.tmap_col,
                           te::SimpleTensor(d_colout, {(size_t)g.rows, (size_t)g.cols}, otype),
                           g.rows, g.cols, RefT::blockIterDim::M, RefT::blockIterDim::N, g.cols, 0,
                           8, RefT::output_swizzle_pattern);

  printf("shape=%dx%d  %s->%s  iters=%d\n", g.rows, g.cols, g.itype_name.c_str(),
         g.otype_name.c_str(), g.iters);
  printf("%-4s %-5s %-5s %10s %10s %8s\n", "ns", "itn", "cvt", "time(ms)", "GB/s", "correct");

  std::vector<Result> results;
#define RUN(NS, ITN, CVT) results.push_back(run_config<NS, ITN, CVT>(g))
  RUN(2, 2, true);
  RUN(2, 4, true);
  RUN(2, 8, true);
  RUN(3, 2, true);
  RUN(3, 4, true);
  RUN(3, 8, true);
  RUN(4, 4, true);
  RUN(2, 4, false);
  RUN(3, 4, false);
#undef RUN

  std::vector<Result> ok;
  for (auto &r : results)
    if (r.ok && r.correct) ok.push_back(r);
  std::sort(ok.begin(), ok.end(), [](const Result &a, const Result &b) { return a.ms < b.ms; });
  if (!ok.empty()) {
    const Result &w = ok.front();
    printf("\nBEST: numStages=%d iterN=%d cvt=%s  %.4f ms  %.1f GB/s\n", w.ns, w.itn,
           w.cvt ? "4x" : "2x", w.ms, w.gbps);
  } else {
    printf("\nno valid config\n");
  }
  return 0;
}
