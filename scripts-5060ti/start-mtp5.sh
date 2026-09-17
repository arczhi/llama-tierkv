#!/usr/bin/env bash
# 5060 Ti 16GB + 32GB RAM tuned launcher for mainline llama.cpp with MTP-5 speculation.
#
#   -ngl 99 --spec-draft-ngl 99   : keep BOTH the target and the MTP head fully on the GPU.
#                                    (do NOT rely on the default --fit: with this model it
#                                    silently offloads layers to the CPU and decode drops
#                                    from ~57 to ~35 tok/s)
#   -c 57344                      : dense KV must fit in VRAM next to the 13.2 GiB model;
#                                    56K is the practical ceiling with q4_0 KV at ub 128.
#   -b 256 -ub 128                : shrinks the MTP draft context's pp graph reserve
#                                    (~400 MiB -> ~290 MiB); non-negotiable at 56K.
#   --spec-draft-n-max 5          : best measured throughput on this card (n=3: 56.6,
#                                    n=5: 63.7 tok/s @8K temp 0; n>=6 acceptance collapses)
#   --spec-draft-type-k/v q4_0    : draft KV is tiny (1 layer); q4_0 saves ~70 MiB
#   --reasoning off               : keep agent output plain (matches the repo's other runs)
#   --temp 0.7 --top-p 0.8 --top-k 20 : Qwen3.8 non-thinking sampling defaults
#
# Measured on RTX 5060 Ti 16GB / Ryzen 9 9950X (2026-09-17):
#   no spec       : 25.7 tok/s decode
#   mtp n=3       : 56.6 tok/s, 73% draft acceptance
#   mtp n=5       : 63.7 tok/s
#   E2E agent task: see references/llama-next-mtp5-experiment.md in the parent project
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf}"
CTX="${CTX:-57344}"
PORT="${PORT:-18210}"
BUILD="${BUILD:-build}"

export LD_LIBRARY_PATH="$(pwd)/$BUILD/bin:${CUDA_LIBS:-/root/tools/cuda-libs}"

exec "$BUILD/bin/llama-server" \
  -m "$MODEL" \
  --alias "${ALIAS:-qwen38-next}" \
  -c "$CTX" \
  --host 127.0.0.1 --port "$PORT" \
  -fa on \
  -ctk q4_0 -ctv q4_0 \
  -b 256 -ub 128 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  -ngl 99 --spec-draft-ngl 99 \
  --parallel 1 --jinja \
  --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 \
  --metrics "$@"
