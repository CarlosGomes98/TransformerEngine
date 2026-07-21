/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Runtime autotuner for the MXFP8 *bidimensional* (rowwise+colwise, TMA) cast
// kernel -- the one dsv3 exercises. For a given shape/dtype it JIT-compiles the
// kernel (NVRTC) for a grid of compile-time tiling configs (pipeline stages x
// per-block iteration width x 4x/2x convert), benchmarks each on-device, checks
// they agree bit-for-bit, and prints a ranking.
//
// Self-contained: it does NOT link libtransformer_engine (whose internal C++
// symbols are hidden by the version script). It builds the TMA descriptors with
// cuTensorMapEncodeTiled and compiles/launches via NVRTC + the CUDA driver API.
// It only *includes* the specialized header (header-only) to read the exact
// BidimTunableTraits geometry/smem so nothing is replicated by hand.
//
// Build (from repo root, in the NGC container) -- fully self-contained, no TE
// headers or lib needed:
//   nvcc -std=c++17 -O3 -o mxfp8_autotune_bidim \
//        benchmarks/mxfp8_autotune_bidimensional.cu -lnvrtc -lcuda
// Run:
//   TE_COMMON=$PWD/transformer_engine/common ./mxfp8_autotune_bidim 4096 7168 bf16 fp8e4m3

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Geometry/smem for the bidimensional (rowwise+colwise) kernel, derived from
// BidimTraitsImpl<IType(2B), OType(1B), NS, ITN, CVT> for the fp16/bf16->fp8
// case (the only case the specialized path takes). Kept in sync with the traits
// in cast/mxfp8/specialized/quantize_mxfp8.cuh; the JIT'd kernel itself uses the
// real traits, so this only has to match its launch parameters.
//   warpDim::num = 1024, warpLayout::num = 2, blockIterDim = {M:32, N:64}
struct Geom {
  int block_x, block_y;      // rowThreadLayout::num, numWarps
  int block_dim_m, block_dim_n;  // blockDIM
  size_t smem;
};
static Geom bidim_geom(int ns, int itn) {
  Geom g;
  g.block_x = 32;                  // rowThreadLayout::num
  g.block_y = 2;                   // numWarps = warpLayout::num
  g.block_dim_m = 32;              // iterLayout::M(1) * blockIterDim::M(32)
  g.block_dim_n = 64 * itn;        // iterLayout::N(itn) * blockIterDim::N(64)
  // smemInput(4096*ns) + smemRowOut(2048*ns) + smemColOut(2048*ns)
  //   + smem_alignment(1024) + smem_rowwise_scale(64*itn) + smem_colwise_reduce(256)
  g.smem = (size_t)8192 * ns + (size_t)64 * itn + 1280;
  return g;
}
// Swizzle enum values (match CUtensorMapSwizzle from <cuda.h>): input 128B, output 64B.

#define CU(x)                                                                 \
  do {                                                                        \
    CUresult r = (x);                                                         \
    if (r != CUDA_SUCCESS) {                                                  \
      const char *s = nullptr;                                                \
      cuGetErrorString(r, &s);                                                \
      fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", r, s, __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)
#define NVRTC(x)                                                              \
  do {                                                                        \
    nvrtcResult r = (x);                                                      \
    if (r != NVRTC_SUCCESS) {                                                 \
      fprintf(stderr, "NVRTC error %s at %s:%d\n", nvrtcGetErrorString(r), __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
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
static std::string replace_all(std::string s, const std::string &a, const std::string &b) {
  size_t p = 0;
  while ((p = s.find(a, p)) != std::string::npos) {
    s.replace(p, a.size(), b);
    p += b.size();
  }
  return s;
}

// Mirror of the library's create_2D_tensor_map (thin cuTensorMapEncodeTiled
// wrapper). type_bits: 16 for bf16/fp16 input, 8 for fp8 output.
static void make_tmap(CUtensorMap *map, void *ptr, int rows, int cols, uint32_t boxM, uint32_t boxN,
                      int type_bits, CUtensorMapSwizzle swz) {
  CUtensorMapDataType dt =
      type_bits == 16 ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8;
  uint64_t size[2] = {(uint64_t)cols, (uint64_t)rows};
  uint64_t stride[1] = {(uint64_t)cols * type_bits / 8};
  uint32_t box[2] = {boxN, boxM};
  uint32_t estride[2] = {1, 1};
  CU(cuTensorMapEncodeTiled(map, dt, 2, ptr, size, stride, box, estride,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

struct Ctx {
  std::string itype, otype, arch;
  int rows, cols, ssr, ssc, iters, reps;
  CUtensorMap tmap_in, tmap_row, tmap_col;
  CUdeviceptr d_sr, d_sc;
  std::string src;
  std::vector<const char *> hsrc, hname;
  std::vector<uint8_t> ref_row, ref_col;
};
struct Result {
  int ns, itn, cvt;
  double ms, gbps, spread;  // ms = min over repeats; spread = (max-min)/min in %
  bool correct, ok;
};

template <int NS, int ITN, bool CVT>
static Result run_config(Ctx &g) {
  const Geom geo = bidim_geom(NS, ITN);
  Result r{NS, ITN, CVT, 0, 0, false, false};

  // --- JIT compile this config ---
  std::string code = replace_all(g.src, "__ITYPE__", g.itype);
  code = replace_all(code, "__OTYPE__", g.otype);
  code = replace_all(code, "__NS__", std::to_string(NS));
  code = replace_all(code, "__ITN__", std::to_string(ITN));
  code = replace_all(code, "__CVT__", CVT ? "true" : "false");
  nvrtcProgram prog;
  NVRTC(nvrtcCreateProgram(&prog, code.c_str(), "autotune_bidim.cu", (int)g.hsrc.size(),
                           g.hsrc.data(), g.hname.data()));
  std::string archopt = "--gpu-architecture=" + g.arch;
  const char *opts[] = {"--std=c++17", archopt.c_str(), "--device-int128", "-default-device",
                        "-I/usr/local/cuda/include"};
  nvrtcResult cr = nvrtcCompileProgram(prog, 5, opts);
  if (cr != NVRTC_SUCCESS) {
    size_t ls;
    nvrtcGetProgramLogSize(prog, &ls);
    std::string log(ls, 0);
    nvrtcGetProgramLog(prog, &log[0]);
    fprintf(stderr, "[ns=%d itn=%d cvt=%d] compile FAILED:\n%s\n", NS, ITN, CVT, log.c_str());
    nvrtcDestroyProgram(&prog);
    return r;
  }
  size_t cbsz;
  NVRTC(nvrtcGetCUBINSize(prog, &cbsz));
  std::vector<char> cubin(cbsz);
  NVRTC(nvrtcGetCUBIN(prog, cubin.data()));
  nvrtcDestroyProgram(&prog);

  CUmodule mod;
  CU(cuModuleLoadData(&mod, cubin.data()));
  CUfunction fn;
  CU(cuModuleGetFunction(&fn, mod, "mxfp8_bidim_autotune_kernel"));

  const size_t smem = geo.smem;
  if (smem > 0) {
    CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
  }
  const unsigned bx = geo.block_x, by = geo.block_y;
  const unsigned gx = (g.cols + geo.block_dim_n - 1) / geo.block_dim_n;
  const unsigned gy = (g.rows + geo.block_dim_m - 1) / geo.block_dim_m;
  void *args[] = {&g.tmap_in, &g.tmap_row, &g.tmap_col, &g.d_sr,  &g.d_sc,
                  &g.rows,    &g.cols,     &g.ssr,       &g.ssc};
  auto launch = [&]() {
    return cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, (unsigned)smem, 0, args, nullptr);
  };

  CUresult lr = launch();
  if (lr != CUDA_SUCCESS) {
    const char *s;
    cuGetErrorString(lr, &s);
    fprintf(stderr, "[ns=%d itn=%d cvt=%d] launch FAILED: %s\n", NS, ITN, CVT, s);
    cuModuleUnload(mod);
    return r;
  }
  CU(cuCtxSynchronize());

  const size_t row_sz = (size_t)g.rows * g.ssr;
  const size_t col_sz = (size_t)((g.rows + 31) / 32) * g.cols;
  std::vector<uint8_t> cur_row(row_sz), cur_col(col_sz);
  CU(cuMemcpyDtoH(cur_row.data(), g.d_sr, row_sz));
  CU(cuMemcpyDtoH(cur_col.data(), g.d_sc, col_sz));
  if (g.ref_row.empty()) {
    g.ref_row = cur_row;
    g.ref_col = cur_col;
    r.correct = true;
  } else {
    r.correct = (cur_row == g.ref_row) && (cur_col == g.ref_col);
  }

  // Time reps independent measurements; report the min (least-noisy) and spread.
  CUevent e0, e1;
  CU(cuEventCreate(&e0, 0));
  CU(cuEventCreate(&e1, 0));
  double ms_min = 1e30, ms_max = 0.0;
  for (int rep = 0; rep < g.reps; rep++) {
    CU(cuEventRecord(e0, 0));
    for (int i = 0; i < g.iters; i++) launch();
    CU(cuEventRecord(e1, 0));
    CU(cuEventSynchronize(e1));
    float total = 0;
    CU(cuEventElapsedTime(&total, e0, e1));
    const double ms = total / g.iters;
    ms_min = ms < ms_min ? ms : ms_min;
    ms_max = ms > ms_max ? ms : ms_max;
  }
  r.ms = ms_min;
  r.spread = ms_min > 0 ? (ms_max - ms_min) / ms_min * 100.0 : 0.0;
  const double bytes = (double)g.rows * g.cols * 2 * 2 + row_sz + col_sz;  // in + 2 outs approx
  r.gbps = bytes / (r.ms * 1e-3) / 1e9;
  r.ok = true;
  printf("%-4d %-5d %-5s %10.4f %9.1f %8.1f%% %8s\n", NS, ITN, CVT ? "4x" : "2x", r.ms, r.gbps,
         r.spread, r.correct ? "yes" : "NO");
  // Machine-parseable per-config line (every config, every shape):
  // CFG <rows> <cols> <itype> <otype> <ns> <itn> <cvt> <ms_min> <gbps> <spread%> <correct 0/1>
  printf("CFG %d %d %s %s %d %d %s %.5f %.1f %.1f %d\n", g.rows, g.cols, g.itype.c_str(),
         g.otype.c_str(), NS, ITN, CVT ? "4x" : "2x", r.ms, r.gbps, r.spread, r.correct ? 1 : 0);
  cuModuleUnload(mod);
  return r;
}

// The tunable bidimensional kernel source (NVRTC). CUtensorMap is the opaque
// struct provided by the included header under __CUDACC_RTC__.
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

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <rows> <cols> <bf16|fp16> <fp8e4m3|fp8e5m2> [iters] [reps]\n",
            argv[0]);
    return 2;
  }
  Ctx g;
  g.rows = atoi(argv[1]);
  g.cols = atoi(argv[2]);
  g.itype = argv[3];
  g.otype = argv[4];
  g.iters = argc > 5 ? atoi(argv[5]) : 50;
  g.reps = argc > 6 ? atoi(argv[6]) : 3;
  g.ssr = (g.cols + 31) / 32;
  g.ssc = g.cols;
  g.src = kSrc;

  const char *common = getenv("TE_COMMON");
  const std::string R = common ? std::string(common) + "/" : "transformer_engine/common/";
  static std::string h0 = read_file(R + "cast/mxfp8/specialized/quantize_mxfp8.cuh");
  static std::string h1 = read_file(R + "util/ptx.cuh");
  static std::string h2 = read_file(R + "cast/mxfp8/specialized/state_counter.cuh");
  static std::string h3 = read_file(R + "cast/mxfp8/specialized/swizzle.cuh");
  static std::string h4 = read_file(R + "utils.cuh");
  static std::string h5 = read_file(R + "util/math.h");
  g.hsrc = {h0.c_str(), h1.c_str(), h2.c_str(), h3.c_str(), h4.c_str(), h5.c_str()};
  g.hname = {"specialized_quantize_mxfp8.cuh", "ptx.cuh", "state_counter.cuh",
             "swizzle.cuh", "utils.cuh", "util/math.h"};

  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;  // primary context: signature is stable across CUDA versions
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  int ccM, ccm;
  CU(cuDeviceGetAttribute(&ccM, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU(cuDeviceGetAttribute(&ccm, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  g.arch = "sm_" + std::to_string(ccM * 10 + ccm) + "a";

  // Device buffers.
  CUdeviceptr d_in, d_rowout, d_colout;
  CU(cuMemAlloc(&d_in, (size_t)g.rows * g.cols * 2));
  CU(cuMemAlloc(&d_rowout, (size_t)g.rows * g.cols));
  CU(cuMemAlloc(&d_colout, (size_t)g.rows * g.cols));
  CU(cuMemAlloc(&g.d_sr, (size_t)g.rows * g.ssr));
  CU(cuMemAlloc(&g.d_sc, (size_t)((g.rows + 31) / 32) * g.cols));
  {
    std::vector<uint16_t> hin((size_t)g.rows * g.cols);
    for (size_t i = 0; i < hin.size(); i++) hin[i] = (uint16_t)((i * 2654435761u) >> 15);
    CU(cuMemcpyHtoD(d_in, hin.data(), hin.size() * 2));
  }

  // Build the 3 TMA descriptors once (box dims / swizzle are config-independent):
  // boxDim = blockIterDim = {M:32, N:64}; input swizzle 128B, output swizzle 64B.
  const uint32_t boxM = 32, boxN = 64;
  make_tmap(&g.tmap_in, (void *)d_in, g.rows, g.cols, boxM, boxN, 16,
            CU_TENSOR_MAP_SWIZZLE_128B);
  make_tmap(&g.tmap_row, (void *)d_rowout, g.rows, g.cols, boxM, boxN, 8,
            CU_TENSOR_MAP_SWIZZLE_64B);
  make_tmap(&g.tmap_col, (void *)d_colout, g.rows, g.cols, boxM, boxN, 8,
            CU_TENSOR_MAP_SWIZZLE_64B);

  printf("shape=%dx%d  %s->%s  device=sm_%d%d  iters=%d\n", g.rows, g.cols, g.itype.c_str(),
         g.otype.c_str(), ccM, ccm, g.iters);
  printf("%-4s %-5s %-5s %10s %9s %9s %8s\n", "ns", "itn", "cvt", "min(ms)", "GB/s", "spread",
         "correct");

  std::vector<Result> results;
#define RUN(NS, ITN, CVT) results.push_back(run_config<NS, ITN, CVT>(g))
  // Reference first = shipped default (numStages=2, iterN=4, 4x); all other
  // configs are checked bit-for-bit against it. cvt fixed at 4x (2x rarely wins
  // for 2-byte input). iterN spans 1..16 to cover occupancy-limited small-N and
  // efficiency-limited large-N regimes.
  RUN(2, 4, true);
  RUN(2, 1, true);
  RUN(2, 2, true);
  RUN(2, 8, true);
  RUN(2, 16, true);
  RUN(3, 1, true);
  RUN(3, 2, true);
  RUN(3, 4, true);
  RUN(3, 8, true);
  RUN(3, 16, true);
  RUN(4, 1, true);
  RUN(4, 2, true);
  RUN(4, 4, true);
  RUN(4, 8, true);
  RUN(4, 16, true);
#undef RUN

  std::vector<Result> ok;
  for (auto &r : results)
    if (r.ok && r.correct) ok.push_back(r);
  std::sort(ok.begin(), ok.end(), [](const Result &a, const Result &b) { return a.ms < b.ms; });
  if (!ok.empty()) {
    const Result &w = ok.front();
    // default (numStages=2, iterN=4, 4x) is the first entry we ran.
    const Result &def = results.front();
    const double speedup = (def.ok && def.correct && w.ms > 0) ? def.ms / w.ms : 1.0;
    printf("\nBEST: numStages=%d iterN=%d cvt=%s  %.4f ms  %.1f GB/s  (%.2fx vs default)\n", w.ns,
           w.itn, w.cvt ? "4x" : "2x", w.ms, w.gbps, speedup);
    // Machine-parseable line for the sweep driver:
    // TUNE <rows> <cols> <itype> <otype> <numStages> <iterN> <cvt> <ms> <gbps> <speedup>
    printf("TUNE %d %d %s %s %d %d %s %.5f %.1f %.3f\n", g.rows, g.cols, g.itype.c_str(),
           g.otype.c_str(), w.ns, w.itn, w.cvt ? "4x" : "2x", w.ms, w.gbps, speedup);
  } else {
    printf("\nno valid config\n");
  }
  return 0;
}
