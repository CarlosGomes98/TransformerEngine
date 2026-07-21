/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Runtime autotuner demo for the MXFP8 rowwise cast-only kernel.
//
// For a given shape/dtype, it JIT-compiles the kernel (via NVRTC) for a grid of
// compile-time tiling configs, benchmarks each on-device, verifies they all
// agree bit-for-bit, and prints a ranking. This is the "figure out compile-time
// options at runtime" idea the NVRTC migration enables: the winning config for a
// shape can then be cached and reused.
//
// Build (from the repo root, in the NGC PyTorch container):
//   nvcc -std=c++17 -O3 -o mxfp8_autotune_rowwise \
//        benchmarks/mxfp8_autotune_rowwise.cu -lnvrtc -lcuda
// Run:
//   TE_COMMON=$PWD/transformer_engine/common ./mxfp8_autotune_rowwise 4096 7168 bf16 fp8e4m3

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CU(x)                                                                   \
  do {                                                                          \
    CUresult r = (x);                                                           \
    if (r != CUDA_SUCCESS) {                                                    \
      const char *s = nullptr;                                                  \
      cuGetErrorString(r, &s);                                                  \
      fprintf(stderr, "CUDA driver error %d (%s) at %s:%d\n", r, s, __FILE__,   \
              __LINE__);                                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

#define NVRTC(x)                                                                \
  do {                                                                          \
    nvrtcResult r = (x);                                                        \
    if (r != NVRTC_SUCCESS) {                                                   \
      fprintf(stderr, "NVRTC error: %s at %s:%d\n", nvrtcGetErrorString(r),     \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

static std::string read_file(const std::string &p) {
  std::ifstream f(p);
  if (!f) {
    fprintf(stderr, "cannot open header %s\n", p.c_str());
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

// The tunable kernel source. extern "C" so the launcher can look it up by name.
static const char *kKernelSrc = R"KSRC(
#include "specialized_quantize_mxfp8.cuh"
using namespace transformer_engine;
namespace specialized = transformer_engine::dispatch::mxfp8::quantize_kernel::specialized;
namespace {
using IType = __ITYPE__;
using OType = __OTYPE__;
using Traits = specialized::RowwiseTunableTraits<IType, OType, __WARP_M__, 1, 1, __USE_CVT_4X__>;
}  // namespace
extern "C" __global__ void __launch_bounds__(Traits::numThreads)
mxfp8_rowwise_autotune_kernel(IType *input, OType *output, unsigned char *scales_rowwise,
                              int rows, int cols, int ssr, int ssc) {
  specialized::quantize_mxfp8_rowwise_cast_only_body<Traits>(
      input, output, scales_rowwise, rows, cols, ssr, ssc);
}
)KSRC";

struct Config {
  int warp_m;
  int use_cvt_4x;
};

struct Result {
  Config cfg;
  double ms;
  double gbps;
  bool correct;
  bool ok;
};

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <rows> <cols> <itype:fp16|bf16> <otype:fp8e4m3|fp8e5m2> [iters]\n",
            argv[0]);
    return 2;
  }
  const int rows = atoi(argv[1]);
  const int cols = atoi(argv[2]);
  const std::string itype = argv[3];
  const std::string otype = argv[4];
  const int iters = argc > 5 ? atoi(argv[5]) : 50;

  const int in_bytes = (itype == "fp16" || itype == "bf16") ? 2 : 4;
  const int out_bytes = 1;  // fp8
  const int ssr = (cols + 31) / 32;  // e8m0 scales per row

  const char *common = getenv("TE_COMMON");
  std::string R = common ? std::string(common) + "/" : "transformer_engine/common/";

  // In-memory headers for NVRTC.
  std::vector<std::pair<std::string, std::string>> H = {
      {"specialized_quantize_mxfp8.cuh", read_file(R + "cast/mxfp8/specialized/quantize_mxfp8.cuh")},
      {"ptx.cuh", read_file(R + "util/ptx.cuh")},
      {"state_counter.cuh", read_file(R + "cast/mxfp8/specialized/state_counter.cuh")},
      {"swizzle.cuh", read_file(R + "cast/mxfp8/specialized/swizzle.cuh")},
      {"utils.cuh", read_file(R + "utils.cuh")},
      {"util/math.h", read_file(R + "util/math.h")},
  };
  std::vector<const char *> hsrc, hname;
  for (auto &h : H) {
    hsrc.push_back(h.second.c_str());
    hname.push_back(h.first.c_str());
  }

  // CUDA context + device buffers.
  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CU(cuCtxCreate(&ctx, 0, dev));
  int cc_major = 0, cc_minor = 0;
  CU(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CU(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  const std::string arch = "sm_" + std::to_string(cc_major * 10 + cc_minor) + "a";

  const size_t in_sz = (size_t)rows * cols * in_bytes;
  const size_t out_sz = (size_t)rows * cols * out_bytes;
  const size_t sc_sz = (size_t)rows * ssr;
  CUdeviceptr d_in, d_out, d_sc;
  CU(cuMemAlloc(&d_in, in_sz));
  CU(cuMemAlloc(&d_out, out_sz));
  CU(cuMemAlloc(&d_sc, sc_sz));

  // Fill input with a deterministic pattern (host).
  std::vector<uint16_t> h_in(in_sz / 2);
  for (size_t i = 0; i < h_in.size(); i++) h_in[i] = (uint16_t)((i * 2654435761u) >> 15);
  CU(cuMemcpyHtoD(d_in, h_in.data(), in_sz));

  CUevent ev0, ev1;
  CU(cuEventCreate(&ev0, 0));
  CU(cuEventCreate(&ev1, 0));

  const std::vector<Config> grid = {{2, 1}, {4, 1}, {8, 1}, {16, 1},
                                    {2, 0}, {4, 0}, {8, 0}, {16, 0}};

  std::vector<uint8_t> ref_out, ref_sc, cur_out(out_sz), cur_sc(sc_sz);
  std::vector<Result> results;

  printf("shape=%dx%d  %s->%s  device=sm_%d%d  arch=%s  iters=%d\n", rows, cols, itype.c_str(),
         otype.c_str(), cc_major, cc_minor, arch.c_str(), iters);
  printf("%-6s %-7s %10s %10s %8s\n", "warp_m", "cvt", "time(ms)", "GB/s", "correct");

  for (size_t ci = 0; ci < grid.size(); ci++) {
    const Config c = grid[ci];
    Result res{c, 0, 0, false, false};

    // --- JIT compile this config ---
    std::string code = replace_all(kKernelSrc, "__ITYPE__", itype);
    code = replace_all(code, "__OTYPE__", otype);
    code = replace_all(code, "__WARP_M__", std::to_string(c.warp_m));
    code = replace_all(code, "__USE_CVT_4X__", c.use_cvt_4x ? "true" : "false");

    nvrtcProgram prog;
    NVRTC(nvrtcCreateProgram(&prog, code.c_str(), "autotune.cu", (int)H.size(), hsrc.data(),
                             hname.data()));
    const std::string archopt = "--gpu-architecture=" + arch;
    const char *opts[] = {"--std=c++17", archopt.c_str(), "--device-int128", "-default-device",
                          "-I/usr/local/cuda/include"};
    nvrtcResult cr = nvrtcCompileProgram(prog, 5, opts);
    if (cr != NVRTC_SUCCESS) {
      size_t ls;
      nvrtcGetProgramLogSize(prog, &ls);
      std::string log(ls, 0);
      nvrtcGetProgramLog(prog, &log[0]);
      fprintf(stderr, "[warp_m=%d cvt4x=%d] compile FAILED:\n%s\n", c.warp_m, c.use_cvt_4x,
              log.c_str());
      results.push_back(res);
      nvrtcDestroyProgram(&prog);
      continue;
    }
    size_t cubin_sz;
    NVRTC(nvrtcGetCUBINSize(prog, &cubin_sz));
    std::vector<char> cubin(cubin_sz);
    NVRTC(nvrtcGetCUBIN(prog, cubin.data()));
    nvrtcDestroyProgram(&prog);

    CUmodule mod;
    CU(cuModuleLoadData(&mod, cubin.data()));
    CUfunction fn;
    CU(cuModuleGetFunction(&fn, mod, "mxfp8_rowwise_autotune_kernel"));

    // --- geometry (mirrors RowwiseTunableTraits) ---
    const int block_dim_n = 1024;         // warpDimN(1024) * warpLayout::N(1) * iterLayout::N(1)
    const int block_dim_m = c.warp_m;     // warpLayout::M * warpDimM(1) * iterLayout::M(1)
    const size_t smem = (size_t)block_dim_m * (block_dim_n / 32) * 1;  // e8m0 scale cache
    if (smem > 0) {
      CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
    }
    void *args[] = {&d_in, &d_out, &d_sc, (void *)&rows, (void *)&cols,
                    (void *)&ssr, (void *)&ssr};
    const unsigned gx = (cols + block_dim_n - 1) / block_dim_n;
    const unsigned gy = (rows + block_dim_m - 1) / block_dim_m;

    auto launch = [&]() {
      return cuLaunchKernel(fn, gx, gy, 1, /*block*/ 32, 1, c.warp_m, (unsigned)smem, 0, args,
                            nullptr);
    };

    // warmup + validate launch
    CUresult lr = launch();
    if (lr != CUDA_SUCCESS) {
      const char *s;
      cuGetErrorString(lr, &s);
      fprintf(stderr, "[warp_m=%d cvt4x=%d] launch FAILED: %s\n", c.warp_m, c.use_cvt_4x, s);
      cuModuleUnload(mod);
      results.push_back(res);
      continue;
    }
    CU(cuCtxSynchronize());

    // correctness vs first successful config
    CU(cuMemcpyDtoH(cur_out.data(), d_out, out_sz));
    CU(cuMemcpyDtoH(cur_sc.data(), d_sc, sc_sz));
    if (ref_out.empty()) {
      ref_out = cur_out;
      ref_sc = cur_sc;
      res.correct = true;
    } else {
      res.correct = (cur_out == ref_out) && (cur_sc == ref_sc);
    }

    // timing
    CU(cuEventRecord(ev0, 0));
    for (int i = 0; i < iters; i++) launch();
    CU(cuEventRecord(ev1, 0));
    CU(cuEventSynchronize(ev1));
    float total_ms = 0;
    CU(cuEventElapsedTime(&total_ms, ev0, ev1));
    res.ms = total_ms / iters;
    const double bytes = (double)in_sz + out_sz + sc_sz;
    res.gbps = bytes / (res.ms * 1e-3) / 1e9;
    res.ok = true;

    printf("%-6d %-7s %10.4f %10.1f %8s\n", c.warp_m, c.use_cvt_4x ? "4x" : "2x", res.ms, res.gbps,
           res.correct ? "yes" : "NO");
    results.push_back(res);
    cuModuleUnload(mod);
  }

  // rank
  std::vector<Result> ok;
  for (auto &r : results)
    if (r.ok && r.correct) ok.push_back(r);
  std::sort(ok.begin(), ok.end(), [](const Result &a, const Result &b) { return a.ms < b.ms; });
  if (!ok.empty()) {
    const Result &w = ok.front();
    printf("\nBEST: warp_m=%d cvt=%s  %.4f ms  %.1f GB/s\n", w.cfg.warp_m,
           w.cfg.use_cvt_4x ? "4x" : "2x", w.ms, w.gbps);
  } else {
    printf("\nno valid config\n");
  }
  return 0;
}
