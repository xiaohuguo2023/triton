# Density-aware `rank_m`: teaching the perf model the difference between real work and padding

*A beginner's walkthrough of the next a8w4 selector optimization.*

This document explains **one specific fix** in plain terms: why the perf model
currently mis-picks GEMM tiles during decode, and how a "density" signal fixes it.
No prior knowledge of the perf model is assumed.

---

## 1. The 30-second version

When the model generates text one token at a time (**decode**), the number we feed
the perf model — `m` — is **mostly padding, not real work**. The perf model can't
tell padding from real work, so it thinks the job is big and picks a **wide tile**
that wastes compute. We want to feed it a signal that reflects *real* work
(**density**) so it picks the right tile on its own — matching JSON where JSON is
good, and beating JSON where JSON's fixed rules are suboptimal.

If none of those words mean anything yet, read on.

---

## 2. Background you need (the setup)

### 2.1 What is the model doing?

gpt-oss-120b is a **Mixture-of-Experts (MoE)** model. Instead of one big feed-forward
network, it has **128 "experts"** (small networks). For each token, a *router* picks
the **top 4** experts to process it. So work is spread unevenly across experts.

### 2.2 What is a "GEMM" and "block_m"?

A **GEMM** is just a matrix multiply — the core operation in the experts. On the GPU
it's computed in **tiles**: the output matrix is chopped into rectangular blocks, and
each block is computed by a group of GPU threads.

- **`block_m`** = the tile height (how many rows of the output one block covers).
- **`block_n`** (or **BN**) = the tile width. Candidates here are 64, 128, 256, 512.

Choosing the tile shape (`block_m` × `block_n`) well is what the perf model does.
`block_m` is decided by the routing (below); the perf model's job is to pick
`block_n`, `block_k`, and a couple of other knobs.

### 2.3 The "grouped" GEMM and where `m` comes from

Because each expert gets a different number of tokens, the experts' matmuls are done
together as a **grouped GEMM**. Think of it as many small matmuls stacked up:

```
expert 0:  M_0 rows  ┐
expert 1:  M_1 rows  │   all stacked into one big "m rows" GEMM,
expert 2:  M_2 rows  │   processed block_m rows at a time
   ...               │
expert E:  M_E rows  ┘
```

`m` = the total number of rows the GEMM processes.

### 2.4 The crucial detail: **padding**

Each expert's rows must be **padded up to a multiple of `block_m`**, because a tile
is `block_m` rows tall and can't be half-used cleanly. So if `block_m = 16` and an
expert only got **1 real token**, it still occupies a **full 16-row block** — 15 rows
of padding, 1 row of real work.

```
expert with 1 real token, block_m=16:

  [ real ]   ← 1 row of actual work
  [ pad  ]
  [ pad  ]     15 rows of padding
   ...
  [ pad  ]   ← still costs a full block
```

---

## 3. Why decode and prefill are completely different

### Prefill (processing the prompt): **dense**
Many tokens arrive at once. Each expert gets *lots* of tokens. Padding is a rounding
error. `m` ≈ real work. Wide tiles (BN256) are great — plenty of work to fill them.

### Decode (generating output): **sparse**
Only a handful of tokens per step. Example from our actual cell (`conc16`):

- 16 tokens per step × top-4 experts = **64 token→expert assignments**
- spread over ~64 different experts → each active expert gets **~1 token**
- with `block_m = 16`, each of those ~64 experts still occupies a **full 16-row block**
- so `m` ≈ 64 experts × 16 rows = **~1024 rows**, but only **~64 rows are real**

**~94% of `m` is padding.** The GEMM *looks* like a 1024-row job but is really a
64-row job scattered across many tiny blocks.

---

## 4. The bug, concretely

The perf model picks a tile by asking, roughly: *"how big is this GEMM (`m`)? Bigger
job → wider tile amortizes better."* Our selector computes a ranking size like
`next_pow2(m)`:

```
decode, block_m=16:   m ≈ 1024 (mostly padding)
                      next_pow2(1024) = 1024
                      model: "1024 rows! big job → use BN256 (wide tile)"
```

But the real work is 64 rows in tiny scattered blocks. A **wide BN256** tile is the
wrong choice — it spreads each tiny block's work too thin and wastes GPU compute on
the padding. We **measured** this (matched-routing A/B on the exact same routing):

| shape (decode) | model picks | measured time/call |
|---|---|---|
| bm16, PM's `next_pow2(m)` | **BN256** | **~18.8 µs**  ← the bug |
| bm16, JSON's heuristic    | **BN64**  | **~6 µs**     ← 3× faster |

JSON avoids the trap not because it's smarter, but because it uses a **fixed rule**:
"small `block_m` → narrow BN." That rule happens to be right for decode.

### 4.1 What our current fix does (and its limit)

The fix we just shipped **caps** the tile: `block_m ≤ 32 → block_n ≤ 128`. This
removes the disastrous BN256. But the model, still fooled by `m ≈ 1024`, now picks
**BN128** (the widest it's *allowed*), while JSON picks **BN64**. So:

```
decode bm16:   BN256 (18.8µs)  →[cap]→  BN128 (10-20µs)  →[goal]→  BN64 (<6µs)
                  the bug             parity-ish            the win
```

The cap got us **out of the ditch (parity)**, but the model is still guessing wrong
*within* the capped choices, because we never fixed **why** it guesses wrong: it's
looking at padded `m`.

---

## 5. The fix: feed the model **density** instead of padded `m`

**Density** = how much real work sits in each block, on average. A simple, cheap
definition using the routing histogram (how many tokens each expert got):

```
density = average blocks per active expert
        = (sum over active experts of ceil(M_e / block_m)) / (number of active experts)
```

What it tells us:

| situation | per-expert tokens | density | meaning |
|---|---|---|---|
| **decode** | ~1 | **≈ 1** | 1 block per expert → sparse, tiny work |
| **prefill** | many | **≫ 1** | many blocks per expert → dense, big work |

**Density separates decode from prefill even when padded `m` looks identical.** That's
the key: `m` can be 1024 in both cases, but density is ~1 for decode and ~13 for
prefill.

### The change to `rank_m`

Instead of always ranking on `next_pow2(m)`:

```python
# today (fooled by padding):
rank_m = next_pow2(m)          # 1024 in decode → BN256/BN128

# proposed (density-aware):
if density is low (sparse / decode-like):
    rank_m = real_work_estimate    # small → model naturally picks BN64
else:
    rank_m = next_pow2(m)          # dense / prefill → wide tiles are correct
```

Now the model **sees a small job in decode and picks the narrow tile itself** — no
cap needed, no hand-written heuristic.

---

## 6. Why this is better than both the cap *and* JSON

- **vs. our cap:** the cap is a blunt "never go wide for small block_m." But sometimes
  small `block_m` *does* have dense real work (a chunked-prefill step can produce
  small `block_m` with many tokens). The cap would wrongly force it narrow. Density
  gets both cases right.

- **vs. JSON:** JSON uses one **fixed rule per block_m**, blind to the actual routing.
  It's right for typical decode but leaves performance on the table when routing is
  unusual. A density-aware model **adapts to the real routing every call** — so it can
  *match* JSON's good decode picks **and beat** JSON where JSON's fixed rule is wrong.
  This is how the "PM is better per-shape" result finally turns into **e2e wins**
  instead of just parity.

In one line: **the cap avoids the model's mistake; density fixes the model's
understanding.**

---

## 7. What it would take to build it

1. **Get the routing histogram at pick time.** We already capture it (the
   `AITER_A8W4_HIST_LOG` code in `perfmodel_a8w4_select.py` reads
   `routing_data.hist` / `.expt_data.hist`). Confirm it's reliably present when
   `pick_a8w4` is called.
2. **Compute density** from the histogram (a few lines: sum of `ceil(M_e/block_m)`
   over nonzero experts, divided by the count of nonzero experts).
3. **Branch `rank_m`** on a density threshold (start simple: sparse if density ≤ ~2).
4. **Validate** exactly as we did the cap:
   - **matched-routing A/B** (`AITER_A8W4_AB`) confirming bm16 now picks BN64 and it's
     faster than BN128;
   - **whole-surface sweep** geomean ≥ the capped run, with decode cells moving from
     parity toward > 1.0;
   - watch that prefill cells (dense) are unchanged (density keeps them on wide tiles).

### Risks / things to watch
- **Threshold calibration:** density is a new tuning knob. Start conservative and let
  the A/B pick the cutoff, rather than guessing a fitted constant.
- **Histogram availability/cost:** if `hist` isn't always present or is expensive to
  read per call, fall back to the cap. (Decode is GPU-bound, so a few µs of host work
  is hidden — but verify.)
- Keep the **cap as a safety net** underneath density until density is proven across
  the surface.

---

## 7.5 FAQ: "Can't we just use a smaller `block_m` (< 16) so there's less padding?"

Natural idea — attack the padding at its source. The honest answer has two layers.

**`block_n` in the *fused* a8w4 MoE kernel: capped at 64.**
- The kernel **fuses swiglu + MXFP4 output re-quantization**, which imposes a hard
  rule: `OUT_BLOCK_N = BLOCK_N // 2 >= 32`, i.e. **`block_n >= 64`**
  (`moe_op_gemm_a8w4.py:186`, assert at `:461`). So this kernel can't go narrower than
  BN64 — that knob isn't exposed. (An unfused decode kernel could; see below.)

**`block_m < 16`: NOT a hardware limit — it's a routing clamp, and it's feasible.**
- Correction to an earlier draft: MFMA does **not** forbid `block_m < 16` (tested).
  The 16 is purely `max(16, …)` in the router (`moe_routing/routing.py:291`). The only
  structural rule is `block_m` must be a **power of two** (`log2_power_of_two`,
  `:216`), so 8/4/2/1 are candidates.
- `block_m` is **not a tuner/perf-model knob** — the router derives it from
  tokens-per-expert (`block_m = routing_data.block_m`, kernel `:90`). To use sub-16
  you change the router clamp; the expt-data/sort infra (`token_offs_pad`,
  `block_pid_map`, `n_blocks`) is already power-of-2-parametric via `block_m_log2`, and
  the MXFP4 microscale swizzle is on the weights (K/N), so it's `block_m`-independent.
- **But in sparse decode the payoff is occupancy, not block count.** Block count is
  expert-limited (≈1 token/expert → 1 block for *any* `block_m`), and the dominant
  traffic is weights (MXFP4), not activations — so a smaller `block_m` only helps if it
  shrinks the accumulator/register footprint → more resident waves → better weight-read
  latency hiding → higher BW. Worth an A/B (block_m=8 vs 16) to size before committing.

**In a *different, unfused* kernel: yes — and it's faster.**
- Standalone GEMM tuners find the decode optimum at **`BLOCK_N=16`** (a single MFMA
  tile) and, for M=1, **`BLOCK_M=1`** via a **GEMV/skinny path** (streaming `.cv`
  cache, no MFMA tiling). vLLM already uses such a kernel (`wvSplitK`) for the small-M
  *dense*/attention path.
- That means the fused a8w4 kernel pays a **"fusion tax"**: decode wants BN16, but
  swiglu+MX-requant fusion forces BN≥64 (~4× wider than ideal). A **narrow-BN or
  GEMV-style a8w4 *decode* kernel** could beat *any* config choice in the current
  kernel — but that's a **kernel-authoring project** (aiter), separate from config
  selection.

**Key points for THIS work (config selection):**
- Our problem was never wasted *padding compute* — it was padded `m` **fooling the
  model into the wrong `block_n`**. Density gives the model the right tile with no
  kernel rewrite.
- The best `block_n` this kernel can do for decode is **BN64** (its floor). So the
  density fix should aim the model at **BN64**; our earlier **cap lands on BN128 —
  provably 2× too wide** vs the achievable BN64. (Bonus: padding shrinks on its own as
  concurrency rises — high conc fills the 16-row blocks. Same density spectrum.)
- The larger decode win (BN16 / GEMV) lives **outside** the selector, in a new kernel.

## 8. TL;DR

- In **decode**, the size number `m` we feed the model is ~94% **padding**.
- The model reads a big `m`, picks a **wide tile (BN256)** → **3× too slow**.
- Our **cap** bans wide tiles for small `block_m` → back to **parity** (but the model
  still picks BN128, not the ideal BN64).
- **Density** (blocks per active expert) tells the model **real work vs padding**, so
  it picks the right tile *itself* → decode goes from **parity to a win**, and PM's
  proven per-shape advantage finally shows up **end-to-end**.

*Related: `../FINDINGS.md` (root-cause + measurements),
`../trace_tools/README.md` (how we measured this),
memory `project_a8w4_pm_vs_json_measured_attribution`.*
