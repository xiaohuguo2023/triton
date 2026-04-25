#include "TritonAMDGPUTransforms/PerfModel.h"
#include <stdio.h>
#include <cmath>
using namespace mlir::triton::AMD::perf;

int main() {
  auto hw = HardwareInfo::get("gfx950");

  // GPT-OSS-120B shape: M=4096, N=5120, K=2880 (N/M=1.25)
  {
    GemmProblem p; p.M=4096; p.N=5120; p.K=2880;
    p.aKind=ElemKind::FP16; p.bKind=ElemKind::FP16; p.cKind=ElemKind::FP32;
    p.aBits=16; p.bBits=16; p.cBits=32;

    auto cands = generateCandidates(p, hw);
    auto ranked = rankConfigs(p, cands, hw, 5);
    printf("M=4096 N=5120 K=2880 (N/M=%.2f)  top-5:\n", (double)p.N/p.M);
    for (int i=0; i<(int)ranked.size(); i++) {
      auto &c = ranked[i]; auto e = estimatePerf(p, c, hw);
      double r = (double)c.blockN/c.blockM;
      printf("  #%d BM=%3d BN=%3d BK=%3d  ratio=%.2f  %.1f TFLOPS\n",
        i+1, c.blockM, c.blockN, c.blockK, r, e.predictedTflops);
    }
  }

  printf("\n");

  // Llama shape: M=4096, N=4096, K=4096 (N/M=1.0, symmetric)
  {
    GemmProblem p; p.M=4096; p.N=4096; p.K=4096;
    p.aKind=ElemKind::FP16; p.bKind=ElemKind::FP16; p.cKind=ElemKind::FP32;
    p.aBits=16; p.bBits=16; p.cBits=32;

    auto cands = generateCandidates(p, hw);
    auto ranked = rankConfigs(p, cands, hw, 3);
    printf("M=4096 N=4096 K=4096 (N/M=%.2f)  top-3:\n", (double)p.N/p.M);
    for (int i=0; i<(int)ranked.size(); i++) {
      auto &c = ranked[i]; auto e = estimatePerf(p, c, hw);
      printf("  #%d BM=%3d BN=%3d BK=%3d  ratio=%.2f  %.1f TFLOPS\n",
        i+1, c.blockM, c.blockN, c.blockK, (double)c.blockN/c.blockM, e.predictedTflops);
    }
  }
}
