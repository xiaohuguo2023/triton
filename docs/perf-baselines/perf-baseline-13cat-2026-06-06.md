# PerfModel Baseline Sweep

Date: 2026-06-06 21:02:01

Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)

| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8192×5120×5120 | LARGE-llama4_maverick | 952 | 965 | 1183 | 0.99× | 0.80× |
| 8192×16384×5120 | LARGE-llama4_maverick | 1008 | 1039 | 1269 | 0.97× | 0.79× |
| 16384×16384×16384 | LARGE-llama3_fp8 | 988 | 1020 | 1288 | 0.97× | 0.77× |
| 4×128×2880 | LARGE_K-gpt_oss_120b | 0 | 0 | 0 | 1.68× | 1.59× |
| 32×128×5120 | LARGE_K-llama4_maverick | 2 | 1 | 2 | 1.93× | 1.00× |
| 256×256×7168 | LARGE_K-deepseek_r1 | 40 | 16 | 43 | 2.56× | 0.93× |
| 64×64×4096 | LARGE_K_SKINNY-synthetic | 2 | 1 | 1 | 1.90× | 2.02× |
| 64×64×8192 | LARGE_K_SKINNY-synthetic | 3 | 1 | 4 | 2.02× | 0.82× |
| 32×64×16384 | LARGE_K_SKINNY-synthetic | 2 | 1 | 4 | 2.16× | 0.53× |
| 32768×128×2880 | LARGE_M-gpt_oss_120b | 489 | 340 | 484 | 1.44× | 1.01× |
| 32768×2880×2880 | LARGE_M-gpt_oss_120b | 918 | 903 | 1094 | 1.02× | 0.84× |
| 4096×128×2880 | LARGE_MK-gpt_oss_120b | 130 | 80 | 119 | 1.63× | 1.09× |
| 16384×128×4096 | LARGE_MK-qwen3_235b_a22b | 375 | 262 | 414 | 1.43× | 0.91× |
| 32768×2112×7168 | LARGE_MK-deepseek_r1 | 874 | 879 | 1001 | 0.99× | 0.87× |
| 4096×32768×512 | LARGE_MN-deepseek_r1 | 646 | 627 | 830 | 1.03× | 0.78× |
| 8192×24576×1536 | LARGE_MN-deepseek_r1 | 909 | 921 | 1115 | 0.99× | 0.82× |
| 32768×24576×1536 | LARGE_MN-deepseek_r1 | 920 | 922 | 1131 | 1.00× | 0.81× |
| 4096×64×64 | LARGE_M_SKINNY-synthetic | 2 | 2 | 2 | 1.58× | 0.99× |
| 8192×64×64 | LARGE_M_SKINNY-synthetic | 5 | 3 | 5 | 1.61× | 0.95× |
| 16384×64×32 | LARGE_M_SKINNY-synthetic | 5 | 3 | 5 | 1.58× | 0.94× |
| 4×32768×512 | LARGE_N-deepseek_r1 | 9 | 6 | 7 | 1.56× | 1.28× |
| 16×24576×1536 | LARGE_N-deepseek_r1 | 33 | 40 | 44 | 0.82× | 0.76× |
| 256×24576×1536 | LARGE_N-deepseek_r1 | 466 | 416 | 504 | 1.12× | 0.92× |
| 1×512×128 | LARGE_NK-deepseek_r1 | 0 | 0 | 0 | 1.63× | 0.97× |
| 16×28672×4096 | LARGE_NK-llama3_mlp | 86 | 73 | 68 | 1.17× | 1.26× |
| 4096×16384×53248 | LARGE_NK-llama3_fp8 | 1019 | 1038 | 1303 | 0.98× | 0.78× |
| 64×4096×64 | LARGE_N_SKINNY-synthetic | 2 | 1 | 2 | 1.65× | 0.97× |
| 64×8192×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.59× | 0.96× |
| 32×16384×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 5 | 1.64× | 1.00× |
| 4096×4096×1536 | MEDIUM-qwen3_235b_a22b | 741 | 699 | 846 | 1.06× | 0.88× |
| 8192×5120×2880 | MEDIUM-gpt_oss_120b | 888 | 916 | 1113 | 0.97× | 0.80× |
| 16384×2112×7168 | MEDIUM-deepseek_r1 | 825 | 873 | 1079 | 0.95× | 0.76× |
| 4096×16384×16384 | MEDIUM-llama3_fp8 | 995 | 1012 | 1256 | 0.98× | 0.79× |
| 64×512×128 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.66× | 1.08× |
| 64×128×512 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.49× | 1.03× |
| 128×128×512 | SMALL-deepseek_r1 | 1 | 1 | 1 | 1.56× | 1.04× |
| 256×128×512 | SMALL-deepseek_r1 | 2 | 2 | 2 | 1.56× | 1.05× |
| 4096×28672×4096 | VERY_LARGE-llama3_mlp | 1007 | 1003 | 1201 | 1.00× | 0.84× |
| 32768×8192×8192 | VERY_LARGE-llama3_fp8 | 1096 | 1063 | 1285 | 1.03× | 0.85× |
| 16384×16384×53248 | VERY_LARGE-llama3_fp8 | 1014 | 1043 | 1298 | 0.97× | 0.78× |

Units: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.
