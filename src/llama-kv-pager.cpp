#include "llama-kv-pager.h"

#include "ggml-backend.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

static bool env_flag(const char * name, bool def = false) {
    const char * v = getenv(name);
    if (!v || !*v) return def;
    return atoi(v) != 0 || strcmp(v, "true") == 0 || strcmp(v, "on") == 0;
}

static uint32_t env_u32(const char * name, uint32_t def) {
    const char * v = getenv(name);
    if (!v || !*v) return def;
    return (uint32_t) strtoul(v, nullptr, 10);
}

void llama_kv_pager::log(const char * fmt, ...) const {
    if (!debug) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}

bool llama_kv_pager::init(const std::vector<layer_ref> & layers_in, size_t k_row_in, size_t v_row_in) {
    enabled = env_flag("LLAMA_KV_PAGER", false);
    if (!enabled) return false;

    max_ctx      = env_u32("LLAMA_KV_PAGER_MAX_CTX", 262144);
    block_tokens = env_u32("LLAMA_KV_PAGER_BLOCK", 64);
    n_seq        = env_u32("LLAMA_KV_PAGER_SEQ", 1);
    stage        = env_flag("LLAMA_KV_PAGER_STAGE", false);
    topk         = env_u32("LLAMA_KV_PAGER_TOPK", 8);
    query_len    = env_u32("LLAMA_KV_PAGER_QUERY", 64);
    debug        = env_flag("LLAMA_KV_PAGER_DEBUG", false);
    pin_max = env_u32("LLAMA_KV_PAGER_PIN_MAX", 512);
    if (const char * p = getenv("LLAMA_KV_PAGER_SAVE")) save_path = p;
    if (const char * p = getenv("LLAMA_KV_PAGER_LOAD")) load_path = p;

    if (block_tokens == 0) block_tokens = 64;
    n_slots = max_ctx * n_seq;

    layers = layers_in;
    k_row = k_row_in;
    v_row = v_row_in;

    k_host.resize(layers.size());
    v_host.resize(layers.size());
    for (size_t il = 0; il < layers.size(); ++il) {
        k_host[il].assign(n_slots * k_row, 0);
        v_host[il].assign(n_slots * v_row, 0);
    }
    slot_saved.assign(n_slots, 0);
    slot_tok.assign(n_slots, -1);
    slot_pinned.assign(n_slots, 0);

    fprintf(stderr, "KV-PAGER: enabled max_ctx=%u block=%u seq=%u stage=%d topk=%u layers=%zu k_row=%zu v_row=%zu store=%.1f MiB\n",
            max_ctx, block_tokens, n_seq, (int) stage, topk, layers.size(), k_row, v_row,
            (k_host.size() * n_slots * (k_row + v_row)) / 1024.0 / 1024.0);
    fflush(stderr);

    if (!load_path.empty()) {
        if (!load_from_file(load_path)) {
            LLAMA_LOG_WARN("%s: failed to load store from '%s'\n", __func__, load_path.c_str());
        }
    }
    return true;
}

int32_t llama_kv_pager::slot_of(llama_seq_id seq, llama_pos pos) const {
    if (seq < 0 || (uint32_t) seq >= n_seq) return -1;
    if (pos < 0 || (uint32_t) pos >= max_ctx) return -1;
    return (int32_t) ((uint32_t) seq * max_ctx + (uint32_t) pos);
}

const uint8_t * llama_kv_pager::peek_k(uint32_t ikv, llama_seq_id seq, llama_pos pos) const {
    if (ikv >= k_host.size()) return nullptr;
    const int32_t slot = slot_of(seq, pos);
    if (slot < 0 || !slot_saved[slot]) return nullptr;
    return k_host[ikv].data() + (size_t) slot * k_row;
}

bool llama_kv_pager::has_row(llama_seq_id seq, llama_pos pos) const {
    const int32_t s = slot_of(seq, pos);
    return s >= 0 && slot_saved[s];
}

bool llama_kv_pager::is_pinned(llama_seq_id seq, llama_pos pos) const {
    const int32_t s = slot_of(seq, pos);
    return s >= 0 && slot_pinned[s];
}

void llama_kv_pager::pin_row(llama_seq_id seq, llama_pos pos) {
    const int32_t s = slot_of(seq, pos);
    if (s < 0) return;
    if (!slot_pinned[s]) {
        slot_pinned[s] = 1;
        pin_fifo.push_back(s);
    }
    // FIFO unpin when over the cap
    while (pin_fifo.size() > pin_max) {
        const int32_t old = pin_fifo.front();
        pin_fifo.erase(pin_fifo.begin());
        slot_pinned[old] = 0;
        log("%s: unpinned slot=%d (fifo)\n", __func__, old);
    }
}

void llama_kv_pager::unpin_row(llama_seq_id seq, llama_pos pos) {
    const int32_t s = slot_of(seq, pos);
    if (s < 0) return;
    slot_pinned[s] = 0;
}

uint64_t llama_kv_pager::n_saved() const {
    uint64_t n = 0;
    for (auto v : slot_saved) n += v;
    return n;
}

void llama_kv_pager::save_row(uint32_t ikv, uint32_t cell, llama_seq_id seq, llama_pos pos, llama_token tok) {
    if (ikv >= layers.size()) return;
    const int32_t slot = slot_of(seq, pos);
    if (slot < 0) return;

    const auto & L = layers[ikv];
    if (L.k && k_row) {
        ggml_backend_tensor_get(L.k, k_host[ikv].data() + (size_t) slot * k_row,
                                (size_t) cell * L.k->nb[1], k_row);
    }
    if (L.v && v_row) {
        ggml_backend_tensor_get(L.v, v_host[ikv].data() + (size_t) slot * v_row,
                                (size_t) cell * L.v->nb[1], v_row);
    }
    slot_saved[slot] = 1;
    slot_tok[slot] = tok;
    ++n_saves;
    log("%s: saved kv layer=%u cell=%u seq=%d pos=%d tok=%d (slot=%d, total=%llu)\n",
        __func__, ikv, cell, seq, pos, tok, slot, (unsigned long long) n_saves);
}

bool llama_kv_pager::load_row(uint32_t ikv, uint32_t cell, llama_seq_id seq, llama_pos pos) {
    if (ikv >= layers.size()) return false;
    const int32_t slot = slot_of(seq, pos);
    if (slot < 0 || !slot_saved[slot]) return false;

    const auto & L = layers[ikv];
    if (L.k && k_row) {
        ggml_backend_tensor_set(L.k, k_host[ikv].data() + (size_t) slot * k_row,
                                (size_t) cell * L.k->nb[1], k_row);
    }
    if (L.v && v_row) {
        ggml_backend_tensor_set(L.v, v_host[ikv].data() + (size_t) slot * v_row,
                                (size_t) cell * L.v->nb[1], v_row);
    }
    ++n_loads;
    return true;
}

std::vector<uint32_t> llama_kv_pager::select_blocks(const llama_token * query, uint32_t n_query) const {
    const uint32_t n_blocks = (max_ctx + block_tokens - 1) / block_tokens;

    std::unordered_set<llama_token> q;
    q.reserve(n_query * 2);
    for (uint32_t i = 0; i < n_query; ++i) {
        if (query[i] >= 0) q.insert(query[i]);
    }

    if (debug) {
        char qbuf[256] = {0};
        for (uint32_t i = 0; i < n_query && i < 8; ++i) {
            char t[24];
            snprintf(t, sizeof(t), " %d", query[n_query - 1 - i]);
            strncat(qbuf, t, sizeof(qbuf) - strlen(qbuf) - 1);
        }
        uint64_t saved_total = n_saved();
        fprintf(stderr, "KV-PAGER selectdbg: nq=%u qtail(last-first):%s | saved=%llu\n",
                n_query, qbuf, (unsigned long long) saved_total);
    }

    static bool hist_dumped = false;
    if (debug && !hist_dumped && n_saved() > 5000) {
        hist_dumped = true;
        std::unordered_map<llama_token, uint32_t> hist;
        for (uint32_t p = 0; p < max_ctx; ++p) {
            const int32_t sl = slot_of(0, (llama_pos) p);
            if (sl >= 0 && slot_saved[sl] && slot_tok[sl] >= 0) {
                hist[slot_tok[sl]]++;
            }
        }
        std::vector<std::pair<uint32_t, llama_token>> top;
        for (auto & kv : hist) top.emplace_back(kv.second, kv.first);
        std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        fprintf(stderr, "KV-PAGER hist: distinct=%zu saved=%llu top:",
                hist.size(), (unsigned long long) n_saved());
        for (size_t i = 0; i < top.size() && i < 5; ++i) {
            fprintf(stderr, " tok%lld x%u", (long long) top[i].second, top[i].first);
        }
        fprintf(stderr, "\n");
    }

    std::vector<std::pair<uint32_t, uint32_t>> scored; // (score, block)
    for (uint32_t b = 0; b < n_blocks; ++b) {
        uint32_t score = 0;
        const uint32_t p0 = b * block_tokens;
        const uint32_t p1 = std::min<uint32_t>(p0 + block_tokens, max_ctx);
        for (uint32_t p = p0; p < p1; ++p) {
            const int32_t s = slot_of(0, (llama_pos) p);
            if (s >= 0 && slot_saved[s] && slot_tok[s] >= 0 && q.count(slot_tok[s])) {
                ++score;
            }
        }
        if (score > 0) scored.emplace_back(score, b);
    }
    std::sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second; // score desc, block asc
    });
    std::vector<uint32_t> res;
    for (size_t i = 0; i < scored.size() && i < topk; ++i) {
        res.push_back(scored[i].second);
    }
    if (debug) {
        fprintf(stderr, "KV-PAGER select: %zu candidate blocks, top:", scored.size());
        for (size_t i = 0; i < res.size(); ++i) {
            fprintf(stderr, " b%u(s%u,tok%lld)", res[i], scored[i].first,
                    (long long) (slot_tok[slot_of(0, (llama_pos) (res[i] * block_tokens))]));
        }
        fprintf(stderr, "\n");
    }
    return res;
}

static bool write_all(FILE * f, const void * p, size_t n) { return fwrite(p, 1, n, f) == n; }
static bool read_all (FILE * f, void * p, size_t n)       { return fread (p, 1, n, f) == n; }

bool llama_kv_pager::save_to_file(const std::string & path) const {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: cannot open '%s' for writing\n", __func__, path.c_str());
        return false;
    }
    const uint32_t magic   = 0x4750564bu; // "KVPG"
    const uint32_t version = 1;
    const uint32_t n_layer = (uint32_t) layers.size();
    const uint64_t kr = k_row, vr = v_row;

    bool ok = true;
    ok &= write_all(f, &magic, 4);
    ok &= write_all(f, &version, 4);
    ok &= write_all(f, &max_ctx, 4);
    ok &= write_all(f, &block_tokens, 4);
    ok &= write_all(f, &n_seq, 4);
    ok &= write_all(f, &n_layer, 4);
    ok &= write_all(f, &kr, 8);
    ok &= write_all(f, &vr, 8);
    ok &= write_all(f, slot_saved.data(), slot_saved.size());
    ok &= write_all(f, slot_tok.data(), slot_tok.size() * sizeof(llama_token));
    for (uint32_t il = 0; il < n_layer && ok; ++il) {
        ok &= write_all(f, k_host[il].data(), k_host[il].size());
        ok &= write_all(f, v_host[il].data(), v_host[il].size());
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_ERROR("%s: write failed for '%s'\n", __func__, path.c_str());
        return false;
    }
    LLAMA_LOG_INFO("%s: saved %llu rows to '%s' (%.1f MiB)\n", __func__,
            (unsigned long long) n_saved(), path.c_str(),
            (double) (n_slots * (k_row + v_row) * layers.size()) / 1024.0 / 1024.0);
    return true;
}

bool llama_kv_pager::load_from_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic = 0, version = 0, n_layer = 0;
    uint64_t kr = 0, vr = 0;
    uint32_t mctx = 0, btok = 0, nseq = 0;
    bool ok = true;
    ok &= read_all(f, &magic, 4);
    ok &= read_all(f, &version, 4);
    ok &= read_all(f, &mctx, 4);
    ok &= read_all(f, &btok, 4);
    ok &= read_all(f, &nseq, 4);
    ok &= read_all(f, &n_layer, 4);
    ok &= read_all(f, &kr, 8);
    ok &= read_all(f, &vr, 8);

    if (!ok || magic != 0x4750564bu || version != 1) {
        LLAMA_LOG_ERROR("%s: '%s' is not a kv-pager store\n", __func__, path.c_str());
        fclose(f);
        return false;
    }
    if (mctx != max_ctx || btok != block_tokens || nseq != n_seq ||
        n_layer != layers.size() || kr != k_row || vr != v_row) {
        LLAMA_LOG_ERROR("%s: store shape mismatch (file: ctx=%u block=%u seq=%u layers=%u k=%llu v=%llu)\n",
                __func__, mctx, btok, nseq, n_layer, (unsigned long long) kr, (unsigned long long) vr);
        fclose(f);
        return false;
    }

    ok &= read_all(f, slot_saved.data(), slot_saved.size());
    ok &= read_all(f, slot_tok.data(), slot_tok.size() * sizeof(llama_token));
    for (uint32_t il = 0; il < n_layer && ok; ++il) {
        ok &= read_all(f, k_host[il].data(), k_host[il].size());
        ok &= read_all(f, v_host[il].data(), v_host[il].size());
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_ERROR("%s: read failed for '%s'\n", __func__, path.c_str());
        return false;
    }
    LLAMA_LOG_INFO("%s: loaded %llu rows from '%s'\n", __func__,
            (unsigned long long) n_saved(), path.c_str());
    return true;
}
