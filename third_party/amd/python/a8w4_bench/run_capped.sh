#!/usr/bin/env bash
# Complete baseline database over the full TP x CONC x shape surface.
# Per cell, runs BOTH arms back-to-back (cold each, same session -> thermal parity):
#   json    : AITER_A8W4_PERFMODEL=0  (tuned dispatch table)
#   pmcanon : AITER_A8W4_PERFMODEL=1  (on-the-fly PerfModel, canonical_m)
# Captures per (arm,cell): stdout, result JSON (latency+throughput), server.log,
# and (PM arm) the per-A8W4-call pick log. Revision/image meta written once.
# PM exact_m is run separately (run_exact.sh) only on losing cells.
#
# Reuses the proven cleanup_container + wait_for_vram (matches this container's
# process names). Cold cache per run (IX-CI parity).
set -uo pipefail

CONTAINER=xguo-nightly-latest
DB=/home/xiaohugu/work/sweep_gptoss_output/pm_baseline_db
REPO=/home/xiaohugu/work/InferenceX
RECIPE=benchmarks/single_node/fixed_seq_len/gptoss_fp4_mi355x.sh
MODEL=/data/amd/gpt-oss-120b-w-mxfp4-a-fp8
RANDOM_RANGE_RATIO=0.8
PORT=8891
mkdir -p "$DB"

TPS=(1 2 4 8)
CONCS=(4 8 16 32 64 128 256)
SHAPES=("8192 1024 9472" "1024 1024 3072")   # prefill-ish, decode-ish

# ---- provenance -------------------------------------------------------------
{
  echo "date_utc=$(docker exec "$CONTAINER" date -u +%FT%TZ)"
  echo "container=$CONTAINER"
  echo "image=$(docker inspect --format '{{.Config.Image}}' "$CONTAINER" 2>/dev/null)"
  echo "recipe=$RECIPE"
  echo "random_range_ratio=$RANDOM_RANGE_RATIO"
  echo "vllm_triton=$(docker exec "$CONTAINER" python3 -c 'import vllm,triton;print(vllm.__version__,triton.__version__)' 2>/dev/null)"
  echo "perf_model_revision=$(docker exec "$CONTAINER" bash -lc 'cd /home/xiaohugu/openai && python3 -c "import perf_model as p;print(p.__perf_model_revision__)"' 2>/dev/null)"
} > "$DB/run_meta.txt"
cat "$DB/run_meta.txt"

cleanup_container() {
  docker exec "$CONTAINER" bash -c '
    pgrep -f "[v]llm serve"            | xargs -r kill -9 2>/dev/null || true
    pgrep -f "/usr/local/bin/vll[m]"   | xargs -r kill -9 2>/dev/null || true
    pgrep -f "[a]pi_server"            | xargs -r kill -9 2>/dev/null || true
    pgrep -f "[E]ngineCore"            | xargs -r kill -9 2>/dev/null || true
    pgrep -f "multiprocessing.[s]pawn" | xargs -r kill -9 2>/dev/null || true
    sleep 6
    rm -rf /root/.cache/vllm/torch_compile_cache/ /root/.triton/ /tmp/torchinductor_root/ 2>/dev/null || true
    mkdir -p /workspace'
}
wait_for_vram() {
  local t=0 gb=999
  while (( t < 50 )); do
    local ub
    ub=$(rocm-smi --showmeminfo VRAM -d 0 2>/dev/null | grep -i "Used Memory" | awk -F: '{print $NF}' | tr -d ' ')
    gb=$(( ${ub:-999999999999} / 1073741824 ))
    (( gb <= 3 )) && return 0
    sleep 3; t=$((t+1))
  done
  echo "  WARN: GPU0 VRAM not drained (${gb} GiB); proceeding"
}

run_arm() { # arm TP CONC ISL OSL MML
  local ARM=$1 TP=$2 CONC=$3 ISL=$4 OSL=$5 MML=$6
  local TAG="${ARM}_isl${ISL}_osl${OSL}_tp${TP}_conc${CONC}"
  local SF="$DB/${TAG}.stdout"
  local pm=0; [[ "$ARM" == pm* ]] && pm=1
  local PICKLOG="$DB/${TAG}.picklog"; rm -f "$PICKLOG"
  echo "== $(date '+%H:%M:%S') $TAG (PERFMODEL=$pm)"
  cleanup_container; wait_for_vram
  docker exec \
    -e MODEL="$MODEL" -e TP="$TP" -e CONC="$CONC" -e ISL="$ISL" -e OSL="$OSL" \
    -e MAX_MODEL_LEN="$MML" -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e RESULT_FILENAME="${TAG}" -e PORT="$PORT" -e RUN_EVAL=false -e EVAL_ONLY=false \
    -e AITER_A8W4_PERFMODEL="$pm" -e PYTHONPATH=/home/xiaohugu/openai \
    -e AITER_A8W4_PM_LOG="$PICKLOG" \
    "$CONTAINER" bash -c "cd $REPO && bash $RECIPE" > "$SF" 2>&1 || echo "  -> non-zero exit"
  # result file is written as <name>.json by the lib; older invocations produced
  # <name>.json.json -- copy whichever exists.
  docker exec "$CONTAINER" bash -c "
    cp -f /workspace/${TAG}.json      ${DB}/${TAG}.result.json 2>/dev/null || \
    cp -f /workspace/${TAG}.json.json ${DB}/${TAG}.result.json 2>/dev/null || true
    cp -f /workspace/server.log       ${DB}/${TAG}.server.log  2>/dev/null || true
    rm -f /workspace/server.log /workspace/gpu_metrics.csv 2>/dev/null || true"
  [[ "$pm" == 0 ]] && rm -f "$PICKLOG"   # JSON arm: no PM picks; shapes come from PM arm's log
  local T O
  T=$(grep -E "Total Token throughput"  "$SF" 2>/dev/null | awk '{print $NF}')
  O=$(grep -E "Output token throughput" "$SF" 2>/dev/null | awk '{print $NF}')
  echo "  total=${T:-FAIL} output=${O:-FAIL} tok/s"
}

echo "== BASELINE DB SWEEP START $(date '+%F %H:%M:%S')  (56 runs: PM-capped, BN<=128 for block_m<=32)"
cleanup_container
for shape in "${SHAPES[@]}"; do
  read -r ISL OSL MML <<< "$shape"
  for TP in "${TPS[@]}"; do
    for CONC in "${CONCS[@]}"; do
      run_arm pmcapped "$TP" "$CONC" "$ISL" "$OSL" "$MML"
    done
  done
done
cleanup_container
echo "== BASELINE DB SWEEP DONE $(date '+%F %H:%M:%S')"
