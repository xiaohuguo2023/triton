#include "TritonAMDGPUTransforms/PerfModel.h"
#include <stdio.h>
using namespace mlir::triton::AMD::perf;
int main() {
  GemmProblem p; p.M=4096; p.N=5120; p.K=2880;
  p.aKind=ElemKind::FP16; p.bKind=ElemKind::FP16; p.cKind=ElemKind::FP32;
  p.aBits=16; p.bBits=16; p.cBits=32;
  auto hw = HardwareInfo::get("gfx950");
  auto cands = generateCandidates(p, hw);
  auto ranked = rankConfigs(p, cands, hw, 3);
  printf("K=2880 (not divisible by 128): top-3 from %zu candidates:\n", cands.size());
  for (int i=0; i<(int)ranked.size(); i++) {
    auto &c = ranked[i]; auto e = estimatePerf(p, c, hw);
    printf("  #%d BM=%3d BN=%3d BK=%3d GSM=%d: %.1f TFLOPS\n",
      i+1, c.blockM, c.blockN, c.blockK, c.groupSizeM, e.predictedTflops);
  }
  printf("\nBK=64 vs BK=128 for 256x128 tile:\n");
  for (int bk : {64, 128}) {
    TritonGemmConfig cfg; cfg.blockM=256; cfg.blockN=128; cfg.blockK=bk;
    cfg.numWarps=8; cfg.numStages=2; cfg.mfmaNonKDim=16;
    cfg.useAsyncCopy=true; cfg.bypassLds=false; cfg.kPack=1; cfg.kWidth=0;
    cfg.groupSizeM=8;
    auto e = estimatePerf(p, cfg, hw);
    double exactIter = (double)p.K / bk;
    printf("  BK=%3d: exact_iter=%.2f  compute=%.0f  TFLOPS=%.1f  valid=%d\n",
      bk, exactIter, e.computeCycles, e.predictedTflops, (int)e.isValid);
  }
}
