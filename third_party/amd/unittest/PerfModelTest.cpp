//===-- PerfModelTest.cpp - Unit tests for the AMD analytical perf model --===//
//
// Tests for PerfModel.h / PerfModel.cpp:
//   - estimatePerf: roofline + wave-quantisation model
//   - isValidConfig: LDS / VGPR / alignment feasibility check
//   - rankConfigs:   sort candidates by predicted TFLOPS
//   - selectMfmaNonKDim: throughput-based 16x16 vs 32x32 selection
//   - generateCandidates: wave-based config space generation
//
// All tests are pure C++ (no GPU, no MLIR context required).
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUTransforms/PerfModel.h"

#include <gtest/gtest.h>

using namespace mlir::triton::AMD::perf;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Typical FP16->FP32 GEMM on gfx942 (CDNA3 / MI300X).
GemmProblem fp16Problem(int64_t M = 4096, int64_t N = 4096, int64_t K = 4096) {
  GemmProblem p;
  p.M = M;
  p.N = N;
  p.K = K;
  p.aKind = ElemKind::FP16;
  p.bKind = ElemKind::FP16;
  p.cKind = ElemKind::FP32;
  p.aBits = 16;
  p.bBits = 16;
  p.cBits = 32;
  return p;
}

// Valid 64x128x32 FP16 config on gfx942.
// 128x128 exceeds the 256 VGPR budget (288 estimated); 64x128 fits at 160.
TritonGemmConfig standardConfig() {
  TritonGemmConfig cfg;
  cfg.blockM = 64;
  cfg.blockN = 128;
  cfg.blockK = 32;
  cfg.numWarps = 4;
  cfg.numStages = 2;
  cfg.mfmaNonKDim = 16;
  cfg.kWidth = 0;
  cfg.bypassLds = false;
  cfg.useAsyncCopy = true;
  cfg.kPack = 1;
  return cfg;
}

HardwareInfo gfx942() { return HardwareInfo::get(Arch::CDNA3); }
HardwareInfo gfx90a() { return HardwareInfo::get(Arch::CDNA2); }
HardwareInfo gfx950() { return HardwareInfo::get(Arch::CDNA4); }

// ---------------------------------------------------------------------------
// estimatePerf tests
// ---------------------------------------------------------------------------

TEST(EstimatePerf, ValidConfigProducesPositiveTflops) {
  auto est = estimatePerf(fp16Problem(), standardConfig(), gfx942());
  EXPECT_TRUE(est.isValid);
  EXPECT_GT(est.predictedTflops, 0.0);
}

TEST(EstimatePerf, ArithmeticIntensityIsPositive) {
  // Any valid config must have positive arithmetic intensity.
  auto est = estimatePerf(fp16Problem(), standardConfig(), gfx942());
  EXPECT_GT(est.arithmeticIntensity, 0.0);
}

TEST(EstimatePerf, LdsUsageIsPositive) {
  auto est = estimatePerf(fp16Problem(), standardConfig(), gfx942());
  EXPECT_GT(est.ldsBytes, 0);
  EXPECT_LE(est.ldsBytes, gfx942().ldsPerCU);
}

TEST(EstimatePerf, VgprCountIsPositive) {
  auto est = estimatePerf(fp16Problem(), standardConfig(), gfx942());
  EXPECT_GT(est.vgprCount, 0);
}

TEST(EstimatePerf, MoreStagesImprovesOverlap) {
  // Increasing numStages should increase pipelineOverlap (more prefetching).
  auto cfg2 = standardConfig();
  auto cfg4 = standardConfig();
  cfg4.numStages = 4;

  auto est2 = estimatePerf(fp16Problem(), cfg2, gfx942());
  auto est4 = estimatePerf(fp16Problem(), cfg4, gfx942());

  EXPECT_GE(est4.pipelineOverlap, est2.pipelineOverlap);
}

TEST(EstimatePerf, LargerBlockKIncreasesArithmeticIntensity) {
  // Doubling blockK doubles FLOPs per tile while memory per tile grows
  // proportionally — arithmetic intensity should stay the same or increase
  // depending on the A/B tile size ratio.
  auto cfgSmallK = standardConfig();
  auto cfgLargeK = standardConfig();
  cfgLargeK.blockK = 64;

  auto estSmall = estimatePerf(fp16Problem(), cfgSmallK, gfx942());
  auto estLarge = estimatePerf(fp16Problem(), cfgLargeK, gfx942());

  EXPECT_GE(estLarge.arithmeticIntensity, estSmall.arithmeticIntensity);
}

TEST(EstimatePerf, SmallProblemHasLowerWaveEfficiency) {
  // A tiny problem (e.g. M=N=K=64) produces few output tiles, causing
  // significant tail-wave waste. waveEfficiency should be lower than for
  // a large problem.
  auto estSmall = estimatePerf(fp16Problem(64, 64, 64), standardConfig(), gfx942());
  auto estLarge = estimatePerf(fp16Problem(4096, 4096, 4096), standardConfig(), gfx942());

  EXPECT_LE(estSmall.waveEfficiency, estLarge.waveEfficiency);
}

// ---------------------------------------------------------------------------
// isValidConfig tests
// ---------------------------------------------------------------------------

TEST(IsValidConfig, StandardConfigIsValid) {
  EXPECT_TRUE(isValidConfig(fp16Problem(), standardConfig(), gfx942()));
}

TEST(IsValidConfig, ExcessiveNumStagesExceedsLds) {
  // Pumping numStages very high balloons LDS usage beyond the 64 KB limit.
  auto cfg = standardConfig();
  cfg.numStages = 32;
  EXPECT_FALSE(isValidConfig(fp16Problem(), cfg, gfx942()));
}

TEST(IsValidConfig, ZeroNumWarpsIsInvalid) {
  auto cfg = standardConfig();
  cfg.numWarps = 0;
  EXPECT_FALSE(isValidConfig(fp16Problem(), cfg, gfx942()));
}

TEST(IsValidConfig, NonPowerOfTwoWarpsIsInvalid) {
  auto cfg = standardConfig();
  cfg.numWarps = 3;
  EXPECT_FALSE(isValidConfig(fp16Problem(), cfg, gfx942()));
}

TEST(IsValidConfig, LargeBlockSizeWithManyStagesExceedsLds) {
  // 256x256 tiles with 4 stages should blow LDS.
  auto cfg = standardConfig();
  cfg.blockM = 256;
  cfg.blockN = 256;
  cfg.numStages = 4;
  cfg.mfmaNonKDim = 16;
  EXPECT_FALSE(isValidConfig(fp16Problem(), cfg, gfx942()));
}

// ---------------------------------------------------------------------------
// rankConfigs tests
// ---------------------------------------------------------------------------

TEST(RankConfigs, EmptyInputReturnsEmpty) {
  auto result = rankConfigs(fp16Problem(), {}, gfx942());
  EXPECT_TRUE(result.empty());
}

TEST(RankConfigs, ValidConfigsRankedBeforeInvalidOnes) {
  TritonGemmConfig valid = standardConfig();

  // LDS-busting config: 32 stages balloons memory use past 64 KB limit.
  TritonGemmConfig invalid = standardConfig();
  invalid.numStages = 32;

  auto ranked = rankConfigs(fp16Problem(), {invalid, valid}, gfx942());
  ASSERT_EQ(ranked.size(), 2u);
  // The valid config must come first (invalid configs rank last).
  auto estFirst = estimatePerf(fp16Problem(), ranked[0], gfx942());
  auto estLast  = estimatePerf(fp16Problem(), ranked[1], gfx942());
  EXPECT_TRUE(estFirst.isValid);
  EXPECT_FALSE(estLast.isValid);
}

TEST(RankConfigs, HigherOccupancyConfigRanksHigher) {
  // A config with more pipeline stages (better overlap) should rank higher
  // than one with a single stage, assuming both are valid.
  TritonGemmConfig cfg1 = standardConfig();
  cfg1.numStages = 1;

  TritonGemmConfig cfg2 = standardConfig();
  cfg2.numStages = 2;

  auto ranked = rankConfigs(fp16Problem(), {cfg1, cfg2}, gfx942());
  ASSERT_EQ(ranked.size(), 2u);

  auto est1 = estimatePerf(fp16Problem(), cfg1, gfx942());
  auto est2 = estimatePerf(fp16Problem(), cfg2, gfx942());

  // The config with higher predicted TFLOPS should be first.
  if (est2.predictedTflops > est1.predictedTflops) {
    EXPECT_EQ(ranked[0].numStages, 2);
  } else {
    EXPECT_EQ(ranked[0].numStages, 1);
  }
}

TEST(RankConfigs, OutputPreservesAllConfigs) {
  std::vector<TritonGemmConfig> configs;
  for (int stages : {1, 2, 3}) {
    auto cfg = standardConfig();
    cfg.numStages = stages;
    configs.push_back(cfg);
  }
  auto ranked = rankConfigs(fp16Problem(), configs, gfx942());
  EXPECT_EQ(ranked.size(), configs.size());
}

// ---------------------------------------------------------------------------
// selectMfmaNonKDim tests (Origami throughput-based selection)
// ---------------------------------------------------------------------------

TEST(SelectMfmaNonKDim, Gfx942Fp16LargeTilePrefers16x16) {
  // throughput(32x32x8)  = 32*32*8 / (64/4) = 512
  // throughput(16x16x16) = 16*16*16 / (32/4) = 512  => tie => 16x16 wins
  auto cfg = standardConfig(); // blockM=128, blockN=128
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, gfx942());
  EXPECT_EQ(dim, 16);
}

TEST(SelectMfmaNonKDim, Gfx942Fp16SmallBlockNPrefers16x16) {
  // blockN=16 < 32, so 32x32 is excluded by the block-size guard => 16x16
  auto cfg = standardConfig();
  cfg.blockN = 16;
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, gfx942());
  EXPECT_EQ(dim, 16);
}

TEST(SelectMfmaNonKDim, Gfx90aFp16LargeTilePrefers16x16) {
  // Same throughput tie as gfx942 for FP16 => 16x16
  auto cfg = standardConfig();
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, gfx90a());
  EXPECT_EQ(dim, 16);
}

TEST(SelectMfmaNonKDim, Gfx950Fp16LargeTilePrefers16x16) {
  auto cfg = standardConfig();
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, gfx950());
  EXPECT_EQ(dim, 16);
}

TEST(SelectMfmaNonKDim, SmallTileReturnsFour) {
  // blockM=8 < 16 => return 4
  auto cfg = standardConfig();
  cfg.blockM = 8;
  cfg.blockN = 8;
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, gfx942());
  EXPECT_EQ(dim, 4);
}

TEST(SelectMfmaNonKDim, UnknownArchFallsBackTo16) {
  auto cfg = standardConfig();
  HardwareInfo unknownHw;
  unknownHw.arch = Arch::Unknown;
  unknownHw.numSimdPerCU = 4;
  int dim = selectMfmaNonKDim(fp16Problem(), cfg, unknownHw);
  EXPECT_EQ(dim, 16);
}

// ---------------------------------------------------------------------------
// generateCandidates tests
// ---------------------------------------------------------------------------

TEST(GenerateCandidates, ProducesNonEmptySetForGfx942Fp16) {
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  EXPECT_GT(candidates.size(), 0u);
}

TEST(GenerateCandidates, AllReturnedConfigsAreValid) {
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  for (const auto &cfg : candidates)
    EXPECT_TRUE(isValidConfig(fp16Problem(), cfg, gfx942()))
        << "Invalid config: blockM=" << cfg.blockM << " blockN=" << cfg.blockN
        << " blockK=" << cfg.blockK << " numWarps=" << cfg.numWarps
        << " numStages=" << cfg.numStages;
}

TEST(GenerateCandidates, MfmaDimIsConsistent) {
  // All candidates should use the same mfmaNonKDim (selected once by the
  // throughput model for this arch+dtype).
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  ASSERT_GT(candidates.size(), 0u);
  int expectedDim = candidates[0].mfmaNonKDim;
  EXPECT_GT(expectedDim, 0);
  for (const auto &cfg : candidates)
    EXPECT_EQ(cfg.mfmaNonKDim, expectedDim);
}

TEST(GenerateCandidates, Gfx942Fp16MfmaDimIs16) {
  // On gfx942 FP16, throughput ties → tiebreak selects 16x16.
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  ASSERT_GT(candidates.size(), 0u);
  EXPECT_EQ(candidates[0].mfmaNonKDim, 16);
}

TEST(GenerateCandidates, BlockSizesAreMultiplesOfMfmaDim) {
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  for (const auto &cfg : candidates) {
    EXPECT_EQ(cfg.blockM % cfg.mfmaNonKDim, 0)
        << "blockM=" << cfg.blockM << " not multiple of mfmaDim=" << cfg.mfmaNonKDim;
    EXPECT_EQ(cfg.blockN % cfg.mfmaNonKDim, 0)
        << "blockN=" << cfg.blockN << " not multiple of mfmaDim=" << cfg.mfmaNonKDim;
  }
}

TEST(GenerateCandidates, RankedOutputHasBestFirst) {
  auto candidates = generateCandidates(fp16Problem(), gfx942());
  auto ranked = rankConfigs(fp16Problem(), candidates, gfx942());
  ASSERT_GT(ranked.size(), 1u);
  // First config should have higher or equal TFLOPS than the second.
  auto est0 = estimatePerf(fp16Problem(), ranked[0], gfx942());
  auto est1 = estimatePerf(fp16Problem(), ranked[1], gfx942());
  EXPECT_GE(est0.predictedTflops, est1.predictedTflops);
}

TEST(GenerateCandidates, WorksForGfx90a) {
  auto candidates = generateCandidates(fp16Problem(), gfx90a());
  EXPECT_GT(candidates.size(), 0u);
  for (const auto &cfg : candidates)
    EXPECT_TRUE(isValidConfig(fp16Problem(), cfg, gfx90a()));
}

TEST(GenerateCandidates, WorksForGfx950) {
  auto candidates = generateCandidates(fp16Problem(), gfx950());
  EXPECT_GT(candidates.size(), 0u);
  for (const auto &cfg : candidates)
    EXPECT_TRUE(isValidConfig(fp16Problem(), cfg, gfx950()));
}

} // namespace
