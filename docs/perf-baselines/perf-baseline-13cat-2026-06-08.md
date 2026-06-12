# PerfModel Baseline Sweep

Date: 2026-06-08 22:13:20

Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)

| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8192×5120×5120 | LARGE-llama4_maverick | 955 | 966 | 1175 | 0.99× | 0.81× |
| 8192×16384×5120 | LARGE-llama4_maverick | 1001 | 1043 | 1267 | 0.96× | 0.79× |
| 16384×16384×16384 | LARGE-llama3_fp8 | 988 | 1021 | 1284 | 0.97× | 0.77× |
| 4×128×2880 | LARGE_K-gpt_oss_120b | 0 | 0 | 0 | 1.66× | 1.71× |
| 32×128×5120 | LARGE_K-llama4_maverick | 3 | 1 | 2 | 2.15× | 1.13× |
| 256×256×7168 | LARGE_K-deepseek_r1 | 44 | 16 | 42 | 2.83× | 1.06× |
| 64×64×4096 | LARGE_K_SKINNY-synthetic | 2 | 1 | 1 | 2.01× | 2.22× |
| 64×64×8192 | LARGE_K_SKINNY-synthetic | 3 | 1 | 3 | 2.25× | 0.95× |
| 32×64×16384 | LARGE_K_SKINNY-synthetic | 2 | 1 | 3 | 2.51× | 0.64× |
| 32768×128×2880 | LARGE_M-gpt_oss_120b | 479 | 341 | 480 | 1.40× | 1.00× |
| 32768×2880×2880 | LARGE_M-gpt_oss_120b | 907 | 896 | 1095 | 1.01× | 0.83× |
| 4096×128×2880 | LARGE_MK-gpt_oss_120b | 134 | 81 | 112 | 1.65× | 1.19× |
| 16384×128×4096 | LARGE_MK-qwen3_235b_a22b | 378 | 262 | 409 | 1.44× | 0.92× |
| 32768×2112×7168 | LARGE_MK-deepseek_r1 | 875 | 878 | 1002 | 1.00× | 0.87× |
| 4096×32768×512 | LARGE_MN-deepseek_r1 | 642 | 625 | 837 | 1.03× | 0.77× |
| 8192×24576×1536 | LARGE_MN-deepseek_r1 | 902 | 907 | 1112 | 0.99× | 0.81× |
| 32768×24576×1536 | LARGE_MN-deepseek_r1 | 924 | 922 | 1128 | 1.00× | 0.82× |
| 4096×64×64 | LARGE_M_SKINNY-synthetic | 2 | 2 | 3 | 1.50× | 0.93× |
| 8192×64×64 | LARGE_M_SKINNY-synthetic | 5 | 3 | 5 | 1.61× | 0.94× |
| 16384×64×32 | LARGE_M_SKINNY-synthetic | 5 | 3 | 5 | 1.60× | 0.94× |
| 4×32768×512 | LARGE_N-deepseek_r1 | 9 | 6 | 7 | 1.58× | 1.28× |
| 16×24576×1536 | LARGE_N-deepseek_r1 | 33 | 40 | 46 | 0.82× | 0.71× |
| 256×24576×1536 | LARGE_N-deepseek_r1 | 471 | 417 | 511 | 1.13× | 0.92× |
| 1×512×128 | LARGE_NK-deepseek_r1 | 0 | 0 | 0 | 1.62× | 0.89× |
| 16×28672×4096 | LARGE_NK-llama3_mlp | 68 | 73 | 69 | 0.94× | 0.99× |
| 4096×16384×53248 | LARGE_NK-llama3_fp8 | 1018 | 1036 | 1306 | 0.98× | 0.78× |
| 64×4096×64 | LARGE_N_SKINNY-synthetic | 2 | 1 | 2 | 1.64× | 0.96× |
| 64×8192×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.60× | 0.93× |
| 32×16384×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.65× | 0.95× |
| 4096×4096×1536 | MEDIUM-qwen3_235b_a22b | 719 | 691 | 905 | 1.04× | 0.79× |
| 8192×5120×2880 | MEDIUM-gpt_oss_120b | 888 | 904 | 1132 | 0.98× | 0.78× |
| 16384×2112×7168 | MEDIUM-deepseek_r1 | 827 | 873 | 1076 | 0.95× | 0.77× |
| 4096×16384×16384 | MEDIUM-llama3_fp8 | 990 | 1010 | 1251 | 0.98× | 0.79× |
| 64×512×128 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.66× | 1.07× |
| 64×128×512 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.57× | 1.09× |
| 128×128×512 | SMALL-deepseek_r1 | 1 | 1 | 1 | 1.55× | 1.09× |
| 256×128×512 | SMALL-deepseek_r1 | 2 | 2 | 2 | 1.54× | 1.10× |
| 4096×28672×4096 | VERY_LARGE-llama3_mlp | 998 | 1003 | 1206 | 1.00× | 0.83× |
| 32768×8192×8192 | VERY_LARGE-llama3_fp8 | 1089 | 1072 | 1280 | 1.02× | 0.85× |
| 16384×16384×53248 | VERY_LARGE-llama3_fp8 | 1017 | 1044 | 1300 | 0.97× | 0.78× |

Units: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.
