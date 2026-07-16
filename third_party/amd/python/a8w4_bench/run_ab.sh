#!/usr/bin/env bash
# Forced-config A/B, single serve: PM selector with AITER_A8W4_AB alternating
# the target no-json shapes between JSON config (A) and PM-bucket config (B)
# call-by-call. One run => both configs measured under IDENTICAL routing / window
# / thermal. The 587 record_function label encodes each returned config, so
# parse disaggregates A vs B. Cell = tp8/conc16/isl1024 (worst-hurt decode cell).
#   A (JSON heuristic bm64) = 128/256/1/4
#   B (PM bucket      bm64) = 256/256/2/8
set -uo pipefail
CONTAINER=xguo-nightly-latest
MODEL=/data/amd/gpt-oss-120b-w-mxfp4-a-fp8
OUT=/home/xiaohugu/work/sweep_gptoss_output/pm_baseline_db/ab_decode
PORT=8892
TP=8; ISL=1024; OSL=1024; MML=3072; CONC=16
NUMP=$(( CONC * 4 ))
COOLDOWN=90
AB="64:1024:3072=128/256/1/4|256/256/2/8;64:3072:512=128/256/1/4|256/256/2/8"
mkdir -p "$OUT"

cleanup() {
  docker exec "$CONTAINER" bash -c '
    pgrep -f "[v]llm serve"|xargs -r kill -9 2>/dev/null || true
    pgrep -f "[E]ngineCore"|xargs -r kill -9 2>/dev/null || true
    pgrep -f "multiprocessing.[s]pawn"|xargs -r kill -9 2>/dev/null || true
    pgrep -f "[a]pi_server"|xargs -r kill -9 2>/dev/null || true
    sleep 6
    rm -rf /root/.cache/vllm/torch_compile_cache/ /root/.triton/ /tmp/torchinductor_root/ 2>/dev/null || true'
}
free_ok() { for _ in $(seq 1 40); do g=$(docker exec "$CONTAINER" python3 -c "import torch;print(int(torch.cuda.mem_get_info(0)[0]/1e9))" 2>/dev/null|tail -1); [[ "${g:-0}" -ge 290 ]] && return 0; sleep 4; done; }

serve() { # dir
  local dir="$1"
  docker exec "$CONTAINER" bash -c "mkdir -p $dir; cat > /workspace/abserve.sh <<EOF
#!/bin/bash
export AMDGCN_USE_BUFFER_OPS=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_ROPE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 HSA_NO_SCRATCH_RECLAIM=1
export AITER_A8W4_PROF_LABEL=1 AITER_A8W4_PERFMODEL=1 PYTHONPATH=/home/xiaohugu/openai:\\\$PYTHONPATH
export AITER_A8W4_AB='$AB'
export AITER_A8W4_PM_LOG=$dir/picklog.txt
exec vllm serve $MODEL --port $PORT --attention-backend ROCM_AITER_UNIFIED_ATTN \
  -cc.pass_config.fuse_rope_kvcache=True --tensor-parallel-size=$TP \
  --gpu-memory-utilization 0.95 --max-model-len $MML --block-size=64 \
  --no-enable-prefix-caching --reasoning-parser openai_gptoss \
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$dir \
  --profiler-config.torch_profiler_use_gzip=true \
  --profiler-config.active_iterations=200 --profiler-config.warmup_iterations=2
EOF
  nohup bash /workspace/abserve.sh > $dir/server.log 2>&1 &"
  for _ in $(seq 1 90); do
    c=$(docker exec "$CONTAINER" bash -c "curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null")
    [[ "$c" == "200" ]] && return 0; sleep 10
  done; echo "  SERVE FAILED"; return 1
}
bench() { # dir
  docker exec "$CONTAINER" bash -c "
    curl -s -X POST http://localhost:$PORT/start_profile >/dev/null 2>&1
    vllm bench serve --backend openai-chat --model $MODEL --endpoint /v1/chat/completions \
      --base-url http://localhost:$PORT --dataset-name random --random-input-len $ISL \
      --random-output-len $OSL --random-range-ratio 0.8 --ignore-eos \
      --num-prompts $NUMP --max-concurrency $CONC 2>/dev/null | awk '/Total token throughput/{print \$NF}'
    curl -s -X POST http://localhost:$PORT/stop_profile >/dev/null 2>&1
    sleep 8"
}

for run in 1 2; do
  d="$OUT/run${run}"; docker exec "$CONTAINER" rm -rf "$d" 2>/dev/null
  echo "== $(date '+%H:%M:%S') run$run (cold; cooldown ${COOLDOWN}s)"
  cleanup; sleep "$COOLDOWN"; free_ok
  serve "$d" || continue
  T=$(bench "$d"); echo "  run$run total=${T:-ERR} tok/s"
  docker exec "$CONTAINER" bash -c "mkdir -p $d/_cpu; mv $d/*async_llm*.pt.trace.json.gz $d/_cpu/ 2>/dev/null; ls $d/*.pt.trace.json.gz 2>/dev/null|wc -l"
done
cleanup
echo "== AB DONE $(date '+%F %H:%M:%S')"
