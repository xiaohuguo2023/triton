#include "TritonAMDGPUTransforms/PerfModel.h"
#include <stdio.h>
using namespace mlir::triton::AMD::perf;

int main() {
  auto hw = HardwareInfo::get("gfx950");
  printf("gfx950: numCUs=%d numXCDs=%d l2SizeBytes=%d\n",
         hw.numCUs, hw.numXCDs, hw.l2SizeBytes);

  for (int M : {768, 1024, 1536, 2048}) {
    GemmProblem p; p.M=M; p.N=M; p.K=M;
    p.aKind=ElemKind::FP16; p.bKind=ElemKind::FP16; p.cKind=ElemKind::FP32;
    p.aBits=16; p.bBits=16; p.cBits=32;

    printf("\nM=%d -- BK=32 vs BK=64 (BM=BN=64, nW=8, nS=2, mfma=16):\n", M);

    for (int bk : {32, 64}) {
      TritonGemmConfig cfg;
      cfg.blockM=64; cfg.blockN=64; cfg.blockK=bk;
      cfg.numWarps=8; cfg.numStages=2; cfg.mfmaNonKDim=16;
      cfg.useAsyncCopy=true; cfg.bypassLds=false; cfg.kPack=1; cfg.kWidth=0;
      cfg.groupSizeM = selectGroupSizeM(p, cfg, hw);

      auto e = estimatePerf(p, cfg, hw);
      int lds = estimateLdsBytes(p, cfg, hw);
      printf("  BK=%2d GSM=%d: valid=%d lds=%6d occ=%.2f "
             "compute=%.0f mem=%.0f eff=%.0f waves=%d "
             "isCompBound=%d TFLOPS=%.1f\n",
        bk, cfg.groupSizeM, e.isValid, lds, e.occupancy,
        e.computeCycles, e.memoryCycles, e.effectiveTileCycles,
        e.numWaves, e.isComputeBound, e.predictedTflops);
    }

    // Also show what generateCandidates + rankConfigs picks
    auto cands = generateCandidates(p, hw);
    auto ranked = rankConfigs(p, cands, hw, 3);
    printf("  Top-3 from model:\n");
    for (int i=0; i<(int)ranked.size(); i++) {
      auto &c = ranked[i]; auto e2 = estimatePerf(p, c, hw);
      printf("    #%d BM=%3d BN=%3d BK=%3d GSM=%d: occ=%.2f %.1f TFLOPS\n",
        i+1, c.blockM, c.blockN, c.blockK, c.groupSizeM,
        e2.occupancy, e2.predictedTflops);
    }
  }
}
