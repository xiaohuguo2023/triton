"""Canonical 24-shape baseline suite for PerfModel-vs-autotune-vs-rocBLAS sweeps.

Three groups:
  A. Square shapes (6)        — Triton tutorial's classic perf curve
  B. Decode / skinny-M (10)    — GPT-OSS 120B layer shapes at varied batch
  C. Prefill / large-batch (8) — GPT-OSS varied-M upper end

Format: (M, N, K, regime_label).
"""

SQUARE = [
    ( 256,  256,  256, "square-tiny"),
    ( 512,  512,  512, "square-small"),
    (1024, 1024, 1024, "square-1k"),
    (2048, 2048, 2048, "square-2k"),
    (3072, 3072, 3072, "square-3k"),
    (4096, 4096, 4096, "square-4k"),
]

DECODE_SKINNY = [
    (   4,  128, 2880, "moe-gating-M4"),
    (  32,  128, 2880, "moe-gating-M32"),
    (   4, 5120, 2880, "input-proj-M4"),
    (  16, 5120, 2880, "input-proj-M16"),
    (  32, 5120, 2880, "input-proj-M32"),
    ( 128, 5120, 2880, "input-proj-M128"),
    (   4, 2880, 4096, "output-proj-M4"),
    (  32, 2880, 4096, "output-proj-M32"),
    ( 128, 2880, 4096, "output-proj-M128"),
    (   4,  640, 2880, "misc-skinny-N640"),
]

PREFILL = [
    ( 4096,   128, 2880, "wide-M4096"),
    ( 4096,  5120, 2880, "input-proj-M4k"),
    ( 8192,  5120, 2880, "input-proj-M8k"),
    ( 4096,  2880, 4096, "output-proj-M4k"),
    ( 8192,  2880, 4096, "output-proj-M8k"),
    (16384,  5120, 2880, "large-prefill-M16k"),
    ( 4096,  4096, 4096, "square-4k-prefill"),
    ( 8192,  8192, 8192, "large-square-8k"),
]

ALL_SHAPES = SQUARE + DECODE_SKINNY + PREFILL

assert len(ALL_SHAPES) == 24, f"expected 24 shapes, got {len(ALL_SHAPES)}"
