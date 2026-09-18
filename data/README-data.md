# Mainline llama.cpp + MTP-5 — raw experiment data (2026-09-17/18)

Companion data file to [llama-next-mtp5-experiment.md](llama-next-mtp5-experiment.md).
All numbers below are extracted from the raw logs shipped in `forks/llama-next/data/`.

## 1. Environment

| Item | Value |
| --- | --- |
| Host | 10.41.3.123, Ubuntu, kernel 6.17.0-41-generic |
| CPU / RAM | AMD Ryzen 9 9950X (16C/32T) / 32GB DDR5-6000 |
| GPU | RTX 5060 Ti 16GB (16,311 MiB), driver 595.71.05, CUDA 13.2 |
| Fork | `ggml-org/llama.cpp` master `c77ae69` (268 commits ahead of the KVMem pin `b81c99b`) |
| Build | Docker `nvidia/cuda:13.2.0-devel-ubuntu24.04` + CUDA 13.2.86 component overlay (nvcc/crt/libnvvm/tileiras), cmake 3.31.6, `GGML_CUDA=ON GGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real`, `LLAMA_CURL=OFF LLAMA_BUILD_TESTS=OFF` |
| Model | `Qwen3.8-27B-UD-IQ4_XS.gguf` (Unsloth) with the 8 MTP matrices requantized to Q4_0 (`Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`, 14,140,820,480 bytes) |

## 2. Spec sweep (8K context, 512 generated tokens, temperature 0)

Raw: `forks/llama-next/data/sweep-results.txt`, `sweep2-results.txt`.
Probe: `forks/llama-next/scripts-5060ti/probe.py` (server-side `timings` from `/v1/chat/completions`).

| Config | Run 1 | Run 2 | Run 3 | Median decode | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| no spec | 25.67 | 25.68 | — | 25.7 | — |
| MTP n=3, draft q8_0, KV q5_0 | 56.61 | 55.89 | — | 56.6 | 73% |
| MTP n=4 | 54.52 | 53.83 | — | 54.4 | — |
| MTP n=5 | 63.67 | 62.10 | — | 63.7 | ~70% |
| MTP n=6 | 47.09 | 45.51 | — | 46.8 | — |
| MTP n=8 | 41.52 | 41.51 | 40.91 | 41.5 | 30.6% |
| MTP n=3 + ngram-mod | 54.48 | 50.36 | — | 54.5 | 56-63% |
| MTP n=5 + ngram-mod | 68.76 | 52.00 | — | 68.8 / 52.0 | 48-65% |
| MTP n=3, draft f16 | 57.05 | 56.75 | — | 57.0 | 73-74% |
| MTP n=3, main KV q4_0 | 52.60 | 52.55 | — | 52.6 | 65% |
| MTP n=5, temp 0.7 | 41.51 | 48.16 | 51.39 | 48.2 | 36-50% |
| MTP n=8, temp 0.7 | 40.39 | 40.16 | 35.98 | 40.2 | 25-30% |

Prefill in all sweep runs: ~840-910 tok/s (except 498 tok/s at ub 128 in the 56K config).

## 3. E2E runs (DeliverableBench ocr-dual-channel)

Raw: `forks/llama-next/data/e2e/` (per-run `meta.json`, `score.json`, `summary.md`, `agent.log`).

| | R1 | R2 | R3 (adopted) |
| --- | ---: | ---: | ---: |
| Server config | 57,344 ctx, q4_0 KV, MTP-5 | same | same + `--presence-penalty 1.5` |
| Thinking | off | off | off |
| Outcome | 100.0 / 100 | looping, killed | **100.0 / 100** |
| Wall time | 865 s (≈600 s agent-side regex stall) | — | **433 s** |
| Tests | 75 passed | — | 71 passed (full green) |
| Tool calls | 62 (bash 45 / read 3 / write 3 / edit 14) | ~295 bash | 61 (bash 33 / read 14 / write 3 / edit 11) |
| Tokens in / out | 39,791 / 13,171 | — | 54,333 / 15,524 |
| Steers | 0 | 1 (manual, ineffective) | 0 |

### R3 server-side statistics (63 generations, 16,143 tokens)

| Metric | Value |
| --- | --- |
| Decode tok/s (p10 / p50 / p90) | **53.4 / 65.8 / 76.6** |
| Decode tok/s (min / max) | 46.1 / **79.7** |
| Draft acceptance overall | **72.8%** (12,678 / 17,420) |
| Mean accepted length per pass | **5.07** |
| Prefill tok/s (p50, ub 128) | 479 |
| Peak VRAM | 15,640 MiB |

### R1 server-side statistics (for contrast)

Decode 46-48 tok/s, draft acceptance 44.6-56.3%, mean len 3.2-3.8, prefill 498-842 tok/s.
Same server config as R3 except the presence penalty — the penalty is what moved
acceptance from ~50% to 72.8% and mean len to 5.07.

## 4. KVMem baseline (2026-09-17, same card/task/agent)

Raw: `forks/llama-next/data/kvmem-server.log` (KVMem server log).

| Metric | Value |
| --- | --- |
| E2E score / wall | 100.0 / 482 s |
| Server decode (151 turns, p50 of per-turn rates) | 50.3 tok/s (min 38.8 / max 65.8) |
| Draft acceptance overall | 83.8% (18,614 / 22,203) |
| Draft config | n_max=3, f16 draft KV, main KV q5_0 |
| Peak context during E2E | 54,255 tokens |
| Peak VRAM / server RSS at 261K | 15,620 MiB / 13.8 GB |

## 5. Method notes

- Decode/prefill tok/s are the **server's own `timings`** (`predicted_per_second`,
  `prompt_per_second`) — same source used by the KVMem comparison.
- E2E wall time = `pi` process lifetime (start of session jsonl to `BENCH_DONE`).
- The 8K sweep prompts are synthetic log text; the E2E numbers reflect real agent traffic
  (tool-call JSON + code), which is why R3 acceptance (72.8%) is higher than the sweep's
  matched-temperature result (36-50%) — structured output is easier to draft.
- R1's wall time includes ~600 s lost to an agent-side bash command stuck in regex
  backtracking (killed); R2 was a looping run (killed after steering). Only R3 is a clean
  completion; both pathologies are agent behavior, not server behavior.

## 6. Raw file index

| File | Content |
| --- | --- |
| `data/sweep-results.txt` | sweep 1 raw output (configs A-G) |
| `data/sweep2-results.txt` | sweep 2 raw output (n_max 4/6/8, temp 0.7) |
| `data/next-server.log` | llama-server log for the R3 run (timings, acceptance, errors) |
| `data/next-nospec.log` | control run log (spec off, 57,344 ctx, q4_0 KV) |
| `data/llama-next-build.log` | full container build log (CUDA 13.2.86 overlay + cmake + ninja) |
| `data/kvmem-server.log` | KVMem fork server log from the 2026-09-17 baseline E2E |
| `data/e2e/llama-next-mtp5{,-r2,-r3}/` | per-run DeliverableBench artifacts |

## 7. Stage-2 profiling and page-sparse prototype

### 7a. Cost model (server, temp 0, ub 128, q5_0 KV)

Fit over n_max = 1/2/3/5 at 8K context: `pass_ms ≈ 40.0 + 5.7 × n_max` (R² > 0.99).

| Config | mean len | decode tok/s | pass ms |
| --- | ---: | ---: | ---: |
| nospec (1 token) | 1.00 | 25.71 | 38.9 |
| n=1 | 1.78 | 38.79 | 46.0 |
| n=2 | 2.19 | 42.36 | 51.7 |
| n=3 | 2.46 | 43.11 | 57.1 |
| n=5 | 2.63 | 38.51 | 68.2 |

Attention scaling (nospec): 38.9 ms/token @8K → 45.5 ms/token @28K → ~0.32 ms per 1K ctx.

Spec knobs with no measurable effect on the per-draft cost: `--no-spec-draft-backend-sampling`,
`--spec-draft-p-split 0`, draft KV f16 vs q8_0 (38.4-39.2 tok/s across all variants).

Note: this synthetic-prose probe under-rates the deployed workload — with `--reasoning off`
the probe's acceptance is 32-78%, while the real agent E2E (R3) reached 72.8% and mean
len 5.07 with the same n_max=5. Tune on agent traffic, not on prose.

### 7b. Page-sparse prototype (`proto/proto_page_sparse.cu`)

256K synthetic KV = 4096 pages × 16 layers × KV dim 1024 (f16; q5_0 would be ~0.44× bytes):

| Stage | ms |
| --- | ---: |
| summary full build (one-time) | 321.6 |
| summary incremental (1 page) | 0.014 |
| scoring 1 row (naive) | 1.83 |
| top-k 1 row (naive) | 3.77 |
| gather K+V from pinned host (67.1 MB) | 2.90 @ 23.2 GB/s |
| naive per-pass total (6 rows) | 36.5 |

Projected with the listed optimizations (score once per layer/page, share top-k across rows,
cache staged pages): **~4-6 ms/pass**. See `references/kv-paging-design.md` §4b.
