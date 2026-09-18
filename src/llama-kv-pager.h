#pragma once

// llama_kv_pager: host-RAM tier for the attention KV cache.
//
// Motivation (see the parent project's references/kv-paging-design.md):
//   dense KV cannot exceed ~57K tokens on a 16 GiB card next to the 13.2 GiB model.
//   The pager saves KV rows that are about to leave the VRAM window into a host buffer
//   (addressable by (seq, pos)), can stage selected rows back into free cells, and can
//   serialize the whole store to disk for cross-restart reuse.
//
// Hooks (in llama-kv-cache.cpp):
//   - apply_ubatch(): before a non-empty cell is overwritten -> save_row
//   - seq_rm():       before cells are removed from a sequence    -> save_row
//   - prepare():      stage selected blocks back into free cells (if LLAMA_KV_PAGER_STAGE=1)
//
// Configuration is via environment variables so the module stays self-contained:
//   LLAMA_KV_PAGER=1              enable (default: off)
//   LLAMA_KV_PAGER_MAX_CTX=N      logical context capacity of the store (default: 262144)
//   LLAMA_KV_PAGER_BLOCK=N        retrieval block size in tokens (default: 64)
//   LLAMA_KV_PAGER_SEQ=N          number of sequences tracked (default: 1)
//   LLAMA_KV_PAGER_STAGE=1        enable retrieval staging (2.4; default: off)
//   LLAMA_KV_PAGER_TOPK=N         blocks staged per prepare() call (default: 8)
//   LLAMA_KV_PAGER_QUERY=N        number of trailing tokens used as the query (default: 64)
//   LLAMA_KV_PAGER_SAVE=PATH      store file for autosave/exit snapshots (2.6)
//   LLAMA_KV_PAGER_AUTOSAVE=N     save the store every N saved rows (default 0 = off)
//   LLAMA_KV_PAGER_LOAD=PATH      load the store from PATH at init (2.6)
//   LLAMA_KV_PAGER_DEBUG=1        verbose logging

#include "ggml.h"
#include "llama.h"

#include <cstddef>

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

struct llama_kv_cache;

struct llama_kv_pager {
    struct layer_ref {
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
    };

    // config
    bool     enabled      = false;
    uint32_t max_ctx      = 262144;
    uint32_t block_tokens = 64;
    uint32_t n_seq        = 1;
    bool     stage        = false;
    uint32_t topk         = 8;
    uint32_t query_len    = 64;
    bool     debug        = false;
    std::string save_path;
    std::string load_path;

    // store
    std::vector<layer_ref> layers;        // one entry per kv layer
    std::vector<std::vector<uint8_t>> k_host; // [layer][slot * k_row]
    std::vector<std::vector<uint8_t>> v_host; // [layer][slot * v_row]
    size_t k_row = 0;                 // host row bytes
    size_t v_row = 0;
    size_t k_row_cache = 0;           // VRAM row bytes (cache type)
    size_t v_row_cache = 0;
    ggml_type type_k_cache = GGML_TYPE_F32;
    ggml_type type_v_cache = GGML_TYPE_F32;
    ggml_type type_k_host  = GGML_TYPE_F32;
    ggml_type type_v_host  = GGML_TYPE_F32;
    int64_t n_embd_k = 0;
    int64_t n_embd_v = 0;
    bool    host_q4 = false;
    uint32_t n_slots = 0;                 // max_ctx * n_seq

    std::vector<uint8_t>     slot_saved;  // [n_slots] 1 if a row is stored
    std::vector<llama_token> slot_tok;    // [n_slots] token id (may be -1)
    std::vector<uint8_t>     slot_pinned; // [n_slots] 1 if staged and protected from eviction
    std::unordered_map<llama_token, uint32_t> tok_freq; // token frequency over stored rows (idf proxy)
    uint64_t                 tok_freq_total = 0;
    std::vector<int32_t>     pin_fifo;    // staged slots in order (for FIFO unpinning)
    uint32_t                 pin_max = 512;
    uint32_t                 autosave_every = 0;
    uint64_t                 last_autosave = 0;

    // stats
    uint64_t n_saves = 0, n_loads = 0, n_stage_calls = 0, n_staged = 0;

    // -- lifecycle ------------------------------------------------------------
    // read env, allocate the host store; returns false if disabled
    bool init(const std::vector<layer_ref> & layers_in, size_t k_row_cache_in, size_t v_row_cache_in,
              ggml_type type_k_cache_in, ggml_type type_v_cache_in, int64_t n_embd_k_in, int64_t n_embd_v_in);

    // -- store ops ------------------------------------------------------------
    // save the row at `cell` of kv-layer `ikv` into host slot (seq,pos)
    void save_row(uint32_t ikv, uint32_t cell, llama_seq_id seq, llama_pos pos, llama_token tok);
    // load host slot (seq,pos) into `cell` of kv-layer `ikv`; returns false if not stored
    bool load_row(uint32_t ikv, uint32_t cell, llama_seq_id seq, llama_pos pos);

    bool     has_row(llama_seq_id seq, llama_pos pos) const;
    const uint8_t * peek_k(uint32_t ikv, llama_seq_id seq, llama_pos pos) const;
    bool     is_pinned(llama_seq_id seq, llama_pos pos) const;
    void     pin_row(llama_seq_id seq, llama_pos pos);
    void     unpin_row(llama_seq_id seq, llama_pos pos);
    int32_t  slot_of  (llama_seq_id seq, llama_pos pos) const;
    uint64_t n_saved() const;

    // -- retrieval (2.4) ------------------------------------------------------
    // score stored blocks (block_tokens-sized) by token overlap with `query`
    // returns up to `topk` (block_index) sorted by score desc
    std::vector<uint32_t> select_blocks(const llama_token * query, uint32_t n_query) const;
    // stage selected rows of one block into the cache's free cells (calls back into cache)
    // implemented in llama-kv-cache.cpp: pager_stage_one(...)

    // -- persistence (2.6) ----------------------------------------------------
    bool save_to_file(const std::string & path) const;
    bool load_from_file(const std::string & path);

    // -- logging --------------------------------------------------------------
    void log(const char * fmt, ...) const;
};
