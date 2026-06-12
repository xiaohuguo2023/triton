# PerfModel Baseline Sweep

Date: 2026-06-12 16:13:56

Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)

| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8192×5120×5120 | LARGE-llama4_maverick | 951 | 980 | 1201 | 0.97× | 0.79× |
| 8192×16384×5120 | LARGE-llama4_maverick | 1005 | 1038 | 1288 | 0.97× | 0.78× |
| 16384×16384×16384 | LARGE-llama3_fp8 | 989 | 1018 | 1293 | 0.97× | 0.77× |
| 4×128×2880 | LARGE_K-gpt_oss_120b | 0 | 0 | 0 | 1.56× | 1.49× |
| 32×128×5120 | LARGE_K-llama4_maverick | 2 | 1 | 2 | 1.87× | 0.96× |
| 256×256×7168 | LARGE_K-deepseek_r1 | 39 | 15 | 41 | 2.52× | 0.95× |
| 64×64×4096 | LARGE_K_SKINNY-synthetic | 2 | 1 | 1 | 1.82× | 1.91× |
| 64×64×8192 | LARGE_K_SKINNY-synthetic | 3 | 1 | 3 | 1.95× | 0.84× |
| 32×64×16384 | LARGE_K_SKINNY-synthetic | 2 | 1 | 3 | 2.11× | 0.55× |
| 32768×128×2880 | LARGE_M-gpt_oss_120b | 458 | 331 | 464 | 1.39× | 0.99× |
| 32768×2880×2880 | LARGE_M-gpt_oss_120b | 912 | 898 | 1120 | 1.02× | 0.81× |
| 4096×128×2880 | LARGE_MK-gpt_oss_120b | 122 | 78 | 113 | 1.57× | 1.08× |
| 16384×128×4096 | LARGE_MK-qwen3_235b_a22b | 361 | 255 | 398 | 1.41× | 0.91× |
| 32768×2112×7168 | LARGE_MK-deepseek_r1 | 916 | 880 | 1017 | 1.04× | 0.90× |
| 4096×32768×512 | LARGE_MN-deepseek_r1 | 630 | 617 | 843 | 1.02× | 0.75× |
| 8192×24576×1536 | LARGE_MN-deepseek_r1 | 894 | 904 | 1121 | 0.99× | 0.80× |
| 32768×24576×1536 | LARGE_MN-deepseek_r1 | 928 | 919 | 1137 | 1.01× | 0.82× |
| 4096×64×64 | LARGE_M_SKINNY-synthetic | 2 | 1 | 2 | 1.50× | 0.94× |
| 8192×64×64 | LARGE_M_SKINNY-synthetic | 4 | 3 | 5 | 1.41× | 0.90× |
| 16384×64×32 | LARGE_M_SKINNY-synthetic | 4 | 3 | 5 | 1.48× | 0.92× |
| 4×32768×512 | LARGE_N-deepseek_r1 | 9 | 6 | 7 | 1.54× | 1.18× |
| 16×24576×1536 | LARGE_N-deepseek_r1 | 32 | 38 | 42 | 0.84× | 0.76× |
| 256×24576×1536 | LARGE_N-deepseek_r1 | 462 | 407 | 486 | 1.14× | 0.95× |
| 1×512×128 | LARGE_NK-deepseek_r1 | 0 | 0 | 0 | 1.46× | 0.88× |
| 16×28672×4096 | LARGE_NK-llama3_mlp | 82 | 74 | 67 | 1.11× | 1.22× |
| 4096×16384×53248 | LARGE_NK-llama3_fp8 | 1022 | 1033 | 1305 | 0.99× | 0.78× |
| 64×4096×64 | LARGE_N_SKINNY-synthetic | 2 | 1 | 2 | 1.48× | 0.96× |
| 64×8192×64 | LARGE_N_SKINNY-synthetic | 4 | 3 | 5 | 1.39× | 0.92× |
| 32×16384×64 | LARGE_N_SKINNY-synthetic | 4 | 3 | 5 | 1.51× | 0.97× |
| 4096×4096×1536 | MEDIUM-qwen3_235b_a22b | 700 | 690 | 863 | 1.01× | 0.81× |
| 8192×5120×2880 | MEDIUM-gpt_oss_120b | 882 | 929 | 1118 | 0.95× | 0.79× |
| 16384×2112×7168 | MEDIUM-deepseek_r1 | 821 | 877 | 1081 | 0.94× | 0.76× |
| 4096×16384×16384 | MEDIUM-llama3_fp8 | 987 | 1012 | 1288 | 0.98× | 0.77× |
| 64×512×128 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.41× | 0.95× |
| 64×128×512 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.51× | 1.06× |
| 128×128×512 | SMALL-deepseek_r1 | 1 | 1 | 1 | 1.53× | 1.06× |
| 256×128×512 | SMALL-deepseek_r1 | 2 | 1 | 2 | 1.51× | 1.05× |
| 4096×28672×4096 | VERY_LARGE-llama3_mlp | 1001 | 1001 | 1231 | 1.00× | 0.81× |
| 32768×8192×8192 | VERY_LARGE-llama3_fp8 | 1088 | 1073 | 1310 | 1.01× | 0.83× |
| 16384×16384×53248 | VERY_LARGE-llama3_fp8 | 1019 | 1038 | 1305 | 0.98× | 0.78× |

Units: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.
