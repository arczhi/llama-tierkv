#!/usr/bin/env bash
# TierKV full-stack agent configuration — the measured E2E config:
#   DeliverableBench ocr-dual-channel, 100.0/100, 398 s, 0 corrections
#   (RTX 5060 Ti 16GB, Qwen3.8-27B IQ4_XS, MTP-5, host-tier KV store + page-sparse selection)
#
# What is active:
#   - pager (TierKV)        VRAM window 49,152 tokens; logical context 65,536
#   - host store            Q4_0 KV rows, lazily allocated (~4.6 GB at 256K logical)
#   - staging               hybrid selector (IDF word overlap + attention page scores)
#   - head protection       first 2,048 positions never evicted (system prompt / task spec)
#   - spec decode           MTP head, n_max=5, Q4_0 draft KV  (agent traffic drafts at 73-80%)
#   - sampling              Qwen3.8 non-thinking defaults + presence penalty 1.5
#                           (the penalty is load-bearing: it suppresses agent loops and
#                            lifts MTP acceptance in structured output)
#
# VRAM note: window 49,152 leaves ~0.4 GiB headroom for CUDA graph instantiation.
# A window equal to the logical context (65,536) crashed on cudaGraphInstantiate here.
#
# Long-context variant (256K logical, no MTP because mainline's draft cannot be windowed):
#   see the README section "Long context" / start with LLAMA_KV_PAGER_WINDOW=32768,
#   LLAMA_KV_PAGER_MAX_CTX=262144, -c 262144 --spec-type none
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf}"
CTX="${CTX:-65536}"
WINDOW="${WINDOW:-49152}"
MAX_CTX="${MAX_CTX:-262144}"
PORT="${PORT:-18210}"
BUILD="${BUILD:-build}"

export LD_LIBRARY_PATH="$(pwd)/$BUILD/bin:${CUDA_LIBS:-/root/tools/cuda-libs}"

export LLAMA_KV_PAGER=1
export LLAMA_KV_PAGER_WINDOW="$WINDOW"
export LLAMA_KV_PAGER_MAX_CTX="$MAX_CTX"
export LLAMA_KV_PAGER_STAGE=1
export LLAMA_KV_PAGER_TOPK=8
export LLAMA_KV_PAGER_QUERY=64
export LLAMA_KV_PAGER_PIN_MAX=4096
export LLAMA_KV_PAGER_HEAD=2048
export LLAMA_KV_PAGER_STAGE_EVERY=1024
export LLAMA_KV_PAGER_SELECT=hybrid
export LLAMA_KV_PAGER_DEBUG="${LLAMA_KV_PAGER_DEBUG:-0}"

exec "$BUILD/bin/llama-server" \
  -m "$MODEL" \
  --alias "${ALIAS:-qwen38-next}" \
  -c "$CTX" \
  --host 127.0.0.1 --port "$PORT" \
  -fa on -ctk q4_0 -ctv q4_0 \
  -b 256 -ub 128 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  -ngl 99 --spec-draft-ngl 99 \
  --parallel 1 --jinja \
  --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 \
  --metrics "$@"
