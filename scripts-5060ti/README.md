# 5060 Ti MTP-5 recipe (mainline llama.cpp)

A tuned single-GPU configuration for **Qwen3.8-27B IQ4_XS + MTP** on an RTX 5060 Ti 16GB,
forked from upstream `ggml-org/llama.cpp` (master c77ae69, 2026-09-17).

## What this is

This is *not* a memory-tiering fork. It is the "spec-decode first" baseline of the
tiering roadmap: prove how far upstream mainline can go on this card when the two
non-obvious traps are avoided, then measure what is left for a KV paging scheme to fix.

## The two traps

1. **`--fit` silently offloads layers to the CPU.** With `-ngl` unset (the default
   `--fit on`), the loader "fits" the model by moving a handful of layers to the CPU
   and decode drops from ~57 to ~35 tok/s. Always pass `-ngl 99 --spec-draft-ngl 99`.
2. **The MTP draft context's prefill graph is oversized.** It inherits `-b/-ub` from
   the main context, so at 56K tokens the draft alone reserves 300-400 MiB of compute
   buffer. `-b 256 -ub 128` shrinks it enough to fit `-c 57344` next to the 13.2 GiB
   model on a 16 GiB card.

## Measured results (2026-09-17, temp 0 probe @ 8K ctx, 512-token generations)

| Configuration | Decode tok/s | Draft acceptance |
| --- | ---: | ---: |
| no speculation | 25.7 | — |
| `draft-mtp` n_max=3 | 56.6 | 73% |
| `draft-mtp` n_max=5 | **63.7** | ~70% |
| `draft-mtp,ngram-mod` n_max=5 | 68.8 / 52.0 (unstable) | 48-65% |
| `draft-mtp` n_max=6/8 | 46.8 / 41.5 | collapses |

E2E (DeliverableBench ocr-dual-channel, `pi` agent, thinking off, 57,344 ctx):
**100.0/100 in 433 s (7.2 min)** with `--presence-penalty 1.5` and MTP-5 — the presence
penalty is not cosmetic: it suppresses the agent's repetition loops and lifts MTP
acceptance from 45-56% to 72.8% (mean accepted length 5.07), giving server-side decode
p50 65.8 / max 79.7 tok/s. See the parent project's
`references/llama-next-mtp5-experiment.md` for the full run table (R1/R2/R3).

## Files

- `start-mtp5.sh` — the tuned launcher (read the header comments for the rationale)
- `probe.py` — decode/prefill/acceptance probe against a running server

## Reproduce

```bash
git checkout <this-fork> # upstream snapshot + this directory
# build (CUDA >= 13.2.86, cmake >= 3.31 for sm_120a):
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j12 --target llama-server
# run:
MODEL=/path/to/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf scripts-5060ti/start-mtp5.sh
```

The IQ4 recipe's model file is Unsloth's `Qwen3.8-27B-UD-IQ4_XS.gguf` with the 8
MTP matrices (`blk.64.*`) requantized to Q4_0 (script: `kvmem-llama.cpp/scripts/quantization/quantize-iq4-mtp.py`).
