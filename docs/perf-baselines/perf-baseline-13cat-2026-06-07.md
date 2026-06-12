# PerfModel Baseline Sweep

Date: 2026-06-07 15:02:34

Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)

| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8192×5120×5120 | LARGE-llama4_maverick | 977 | 952 | 1187 | 1.03× | 0.82× |
| 8192×16384×5120 | LARGE-llama4_maverick | 1005 | 1038 | 1269 | 0.97× | 0.79× |
| 16384×16384×16384 | LARGE-llama3_fp8 | 989 | 1020 | 1293 | 0.97× | 0.77× |
| 4×128×2880 | LARGE_K-gpt_oss_120b | 0 | 0 | 0 | 1.66× | 1.72× |
| 32×128×5120 | LARGE_K-llama4_maverick | 3 | 1 | 2 | 2.15× | 1.11× |
| 256×256×7168 | LARGE_K-deepseek_r1 | 45 | 16 | 44 | 2.88× | 1.03× |
| 64×64×4096 | LARGE_K_SKINNY-synthetic | 2 | 1 | 1 | 2.06× | 2.23× |
| 64×64×8192 | LARGE_K_SKINNY-synthetic | 3 | 1 | 4 | 2.23× | 0.91× |
| 32×64×16384 | LARGE_K_SKINNY-synthetic | 2 | 1 | 4 | 2.52× | 0.63× |
| 32768×128×2880 | LARGE_M-gpt_oss_120b | 475 | 335 | 467 | 1.42× | 1.02× |
| 32768×2880×2880 | LARGE_M-gpt_oss_120b | 908 | 897 | 1099 | 1.01× | 0.83× |
| 4096×128×2880 | LARGE_MK-gpt_oss_120b | 132 | 81 | 116 | 1.64× | 1.14× |
| 16384×128×4096 | LARGE_MK-qwen3_235b_a22b | 381 | 261 | 392 | 1.46× | 0.97× |
| 32768×2112×7168 | LARGE_MK-deepseek_r1 | 871 | 876 | 1001 | 1.00× | 0.87× |
| 4096×32768×512 | LARGE_MN-deepseek_r1 | 641 | 631 | 823 | 1.02× | 0.78× |
| 8192×24576×1536 | LARGE_MN-deepseek_r1 | 902 | 925 | 1117 | 0.98× | 0.81× |
| 32768×24576×1536 | LARGE_MN-deepseek_r1 | 924 | 921 | 1126 | 1.00× | 0.82× |
| 4096×64×64 | LARGE_M_SKINNY-synthetic | 2 | 1 | 2 | 1.63× | 1.03× |
| 8192×64×64 | LARGE_M_SKINNY-synthetic | 5 | 3 | 5 | 1.62× | 1.04× |
| 16384×64×32 | LARGE_M_SKINNY-synthetic | 5 | 3 | 4 | 1.61× | 1.09× |
| 4×32768×512 | LARGE_N-deepseek_r1 | 9 | 6 | 8 | 1.60× | 1.20× |
| 16×24576×1536 | LARGE_N-deepseek_r1 | 33 | 40 | 46 | 0.83× | 0.72× |
| 256×24576×1536 | LARGE_N-deepseek_r1 | 464 | 412 | 520 | 1.13× | 0.89× |
| 1×512×128 | LARGE_NK-deepseek_r1 | 0 | 0 | 0 | 1.67× | 0.89× |
| 16×28672×4096 | LARGE_NK-llama3_mlp | 70 | 74 | 69 | 0.95× | 1.01× |
| 4096×16384×53248 | LARGE_NK-llama3_fp8 | 1017 | 1036 | 1303 | 0.98× | 0.78× |
| 64×4096×64 | LARGE_N_SKINNY-synthetic | 2 | 1 | 2 | 1.65× | 0.98× |
| 64×8192×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.58× | 0.96× |
| 32×16384×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.69× | 1.07× |
| 4096×4096×1536 | MEDIUM-qwen3_235b_a22b | 744 | 699 | 869 | 1.07× | 0.86× |
| 8192×5120×2880 | MEDIUM-gpt_oss_120b | 892 | 911 | 1121 | 0.98× | 0.80× |
| 16384×2112×7168 | MEDIUM-deepseek_r1 | 819 | 871 | 1087 | 0.94× | 0.75× |
| 4096×16384×16384 | MEDIUM-llama3_fp8 | 986 | 1012 | 1250 | 0.97× | 0.79× |
| 64×512×128 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.63× | 1.09× |
| 64×128×512 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.52× | 1.08× |
| 128×128×512 | SMALL-deepseek_r1 | 1 | 1 | 1 | 1.55× | 1.11× |
| 256×128×512 | SMALL-deepseek_r1 | 2 | 2 | 2 | 1.56× | 1.10× |
| 4096×28672×4096 | VERY_LARGE-llama3_mlp | 1003 | 1006 | 1202 | 1.00× | 0.83× |
| 32768×8192×8192 | VERY_LARGE-llama3_fp8 | 1089 | 1059 | 1279 | 1.03× | 0.85× |
| 16384×16384×53248 | VERY_LARGE-llama3_fp8 | 1016 | 1043 | 1298 | 0.97× | 0.78× |

Units: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.
