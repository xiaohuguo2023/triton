# PerfModel Baseline Sweep

Date: 2026-06-09 11:32:39

Backends: PerfModel pick (top-1) · Triton autotune (18 configs) · rocBLAS (torch.matmul)

| shape (M×N×K) | regime | PM | autotune | rocBLAS | PM/auto | PM/rocBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8192×5120×5120 | LARGE-llama4_maverick | 895 | 934 | 1166 | 0.96× | 0.77× |
| 8192×16384×5120 | LARGE-llama4_maverick | 959 | 959 | 778 | 1.00× | 1.23× |
| 16384×16384×16384 | LARGE-llama3_fp8 | 892 | 906 | 722 | 0.98× | 1.23× |
| 4×128×2880 | LARGE_K-gpt_oss_120b | 0 | 0 | 0 | 1.64× | 1.61× |
| 32×128×5120 | LARGE_K-llama4_maverick | 2 | 1 | 2 | 1.95× | 1.07× |
| 256×256×7168 | LARGE_K-deepseek_r1 | 40 | 16 | 40 | 2.58× | 0.99× |
| 64×64×4096 | LARGE_K_SKINNY-synthetic | 2 | 1 | 1 | 1.90× | 2.20× |
| 64×64×8192 | LARGE_K_SKINNY-synthetic | 3 | 1 | 3 | 2.01× | 0.87× |
| 32×64×16384 | LARGE_K_SKINNY-synthetic | 2 | 1 | 3 | 2.16× | 0.59× |
| 32768×128×2880 | LARGE_M-gpt_oss_120b | 479 | 341 | 474 | 1.41× | 1.01× |
| 32768×2880×2880 | LARGE_M-gpt_oss_120b | 821 | 853 | 703 | 0.96× | 1.17× |
| 4096×128×2880 | LARGE_MK-gpt_oss_120b | 133 | 80 | 109 | 1.65× | 1.22× |
| 16384×128×4096 | LARGE_MK-qwen3_235b_a22b | 375 | 261 | 378 | 1.44× | 0.99× |
| 32768×2112×7168 | LARGE_MK-deepseek_r1 | 858 | 865 | 980 | 0.99× | 0.88× |
| 4096×32768×512 | LARGE_MN-deepseek_r1 | 558 | 595 | 502 | 0.94× | 1.11× |
| 8192×24576×1536 | LARGE_MN-deepseek_r1 | 865 | 845 | 674 | 1.02× | 1.28× |
| 32768×24576×1536 | LARGE_MN-deepseek_r1 | 898 | 882 | 622 | 1.02× | 1.44× |
| 4096×64×64 | LARGE_M_SKINNY-synthetic | 2 | 1 | 2 | 1.66× | 1.11× |
| 8192×64×64 | LARGE_M_SKINNY-synthetic | 5 | 2 | 4 | 2.15× | 1.16× |
| 16384×64×32 | LARGE_M_SKINNY-synthetic | 5 | 3 | 4 | 1.57× | 1.10× |
| 4×32768×512 | LARGE_N-deepseek_r1 | 9 | 6 | 7 | 1.63× | 1.33× |
| 16×24576×1536 | LARGE_N-deepseek_r1 | 34 | 40 | 43 | 0.85× | 0.77× |
| 256×24576×1536 | LARGE_N-deepseek_r1 | 480 | 411 | 484 | 1.17× | 0.99× |
| 1×512×128 | LARGE_NK-deepseek_r1 | 0 | 0 | 0 | 1.68× | 1.03× |
| 16×28672×4096 | LARGE_NK-llama3_mlp | 86 | 73 | 67 | 1.17× | 1.27× |
| 4096×16384×53248 | LARGE_NK-llama3_fp8 | 871 | 934 | 716 | 0.93× | 1.22× |
| 64×4096×64 | LARGE_N_SKINNY-synthetic | 2 | 1 | 2 | 1.68× | 1.11× |
| 64×8192×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 4 | 1.67× | 1.09× |
| 32×16384×64 | LARGE_N_SKINNY-synthetic | 5 | 3 | 4 | 1.63× | 1.06× |
| 4096×4096×1536 | MEDIUM-qwen3_235b_a22b | 494 | 677 | 593 | 0.73× | 0.83× |
| 8192×5120×2880 | MEDIUM-gpt_oss_120b | 866 | 893 | 1116 | 0.97× | 0.78× |
| 16384×2112×7168 | MEDIUM-deepseek_r1 | 826 | 869 | 1098 | 0.95× | 0.75× |
| 4096×16384×16384 | MEDIUM-llama3_fp8 | 881 | 904 | 738 | 0.97× | 1.19× |
| 64×512×128 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.65× | 1.07× |
| 64×128×512 | SMALL-deepseek_r1 | 1 | 0 | 1 | 1.77× | 1.10× |
| 128×128×512 | SMALL-deepseek_r1 | 1 | 1 | 1 | 1.51× | 1.08× |
| 256×128×512 | SMALL-deepseek_r1 | 2 | 2 | 2 | 1.53× | 1.12× |
| 4096×28672×4096 | VERY_LARGE-llama3_mlp | 949 | 948 | 780 | 1.00× | 1.22× |
| 32768×8192×8192 | VERY_LARGE-llama3_fp8 | 966 | 976 | 700 | 0.99× | 1.38× |
| 16384×16384×53248 | VERY_LARGE-llama3_fp8 | 906 | 928 | 737 | 0.98× | 1.23× |

Units: TFLOPS. Ratios > 1 means PerfModel pick beats the reference.
