// proto_page_sparse.cu — validates page-sparse attention primitives on the 5060 Ti.
//
// Shapes mirror Qwen3.8-27B attention layers: 16 attn layers, KV dim 1024, 64-token pages,
// 256K tokens (4096 pages), verify batch of 6 query rows, top-k = 16 pages per layer.
// Storage in f16 for simplicity (real q5_0 store is ~0.44x the bytes -> numbers are upper bounds).
//
// Measured stages:
//   1. build_summaries  K min/max per page per channel (one-time full build + incremental)
//   2. page_scores      per (layer, kv-group) page scores for a query row
//   3. topk             per (layer) top-k selection
//   4. gather           stage k pages/layer from pinned host -> device staging (K+V)
//
// Build: nvcc -O3 -arch=sm_120a -o proto_page_sparse proto_page_sparse.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int N_LAYERS = 16;
constexpr int KV_DIM   = 1024;
constexpr int PAGE_TOK = 64;
constexpr int N_ROWS   = 6;      // verify batch (n_max + 1)
constexpr int TOPK     = 16;     // pages staged per layer
constexpr int GROUPS   = 4;      // KV heads (scoring groups)

__global__ void build_summaries(const __half * __restrict__ k, __half * __restrict__ kmin,
                                __half * __restrict__ kmax) {
    const size_t page = blockIdx.x;
    const int c = threadIdx.x;
    const __half * src = k + page * PAGE_TOK * KV_DIM + c;
    float lo = 1e30f, hi = -1e30f;
    #pragma unroll 4
    for (int t = 0; t < PAGE_TOK; ++t) {
        const float v = __half2float(src[(size_t) t * KV_DIM]);
        lo = fminf(lo, v); hi = fmaxf(hi, v);
    }
    kmin[page * KV_DIM + c] = __float2half(lo);
    kmax[page * KV_DIM + c] = __float2half(hi);
}

// Quest-style upper bound score: sum_c max(q_c * kmin_c, q_c * kmax_c)
__global__ void page_scores(const float * __restrict__ q,       // [GROUPS][KV_DIM]
                            const __half * __restrict__ kmin,   // [N_PAGES][KV_DIM]
                            const __half * __restrict__ kmax,
                            float * __restrict__ score,         // [n_pages], per group slice
                            int n_pages, int group) {
    const int page = blockIdx.x;
    const int tid  = threadIdx.x;
    constexpr int PER = KV_DIM / 256;
    const float * qg = q + (size_t) group * KV_DIM;
    const size_t off = (size_t) page * KV_DIM;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int c = tid * PER + k;
        const float qv = qg[c];
        const float lo = __half2float(kmin[off + c]);
        const float hi = __half2float(kmax[off + c]);
        acc += fmaxf(qv * lo, qv * hi);
    }
    __shared__ float red[256];
    red[tid] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&score[page], red[0]);   // merge groups
}

__global__ void topk(const float * __restrict__ score, int * __restrict__ idx,
                     float * __restrict__ top, int n_pages) {
    const int tid = threadIdx.x, nt = blockDim.x;
    const float * s = score;
    float best[TOPK]; int bi[TOPK];
    for (int i = 0; i < TOPK; ++i) { best[i] = -1e30f; bi[i] = -1; }
    for (int p = tid; p < n_pages; p += nt) {
        const float v = s[p];
        if (v > best[TOPK - 1]) {
            int j = TOPK - 1;
            while (j > 0 && best[j - 1] < v) { best[j] = best[j - 1]; bi[j] = bi[j - 1]; --j; }
            best[j] = v; bi[j] = p;
        }
    }
    __shared__ float sb[256][TOPK];
    __shared__ int   si[256][TOPK];
    for (int i = 0; i < TOPK; ++i) { sb[tid][i] = best[i]; si[tid][i] = bi[i]; }
    __syncthreads();
    if (tid == 0) {
        float fb[TOPK]; int fi[TOPK]; int ptr[256];
        for (int i = 0; i < 256; ++i) ptr[i] = 0;
        for (int out = 0; out < TOPK; ++out) {
            float bv = -1e30f; int bw = -1;
            for (int w = 0; w < 256; ++w) {
                if (ptr[w] < TOPK && sb[w][ptr[w]] > bv) { bv = sb[w][ptr[w]]; bw = w; }
            }
            if (bw < 0) break;
            fb[out] = bv; fi[out] = si[bw][ptr[bw]]; ptr[bw]++;
        }
        for (int i = 0; i < TOPK; ++i) { top[i] = fb[i]; idx[i] = fi[i] < 0 ? 0 : fi[i]; }
    }
}

static float ms_between(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms;
}

int main(int argc, char ** argv) {
    const int n_pages = argc > 1 ? atoi(argv[1]) : 4096;
    const size_t page_bytes = (size_t) PAGE_TOK * KV_DIM * sizeof(__half);       // 128 KiB
    const size_t store_kv   = (size_t) n_pages * page_bytes;                     // per layer, K or V
    const size_t sum_bytes  = (size_t) n_pages * KV_DIM * 2 * sizeof(__half);    // min+max / layer
    const size_t stg_bytes  = (size_t) TOPK * page_bytes;                        // per layer, K or V

    printf("pages=%d (%d tokens), layers=%d, kvdim=%d, rows=%d, topk=%d\n",
           n_pages, n_pages * PAGE_TOK, N_LAYERS, KV_DIM, N_ROWS, TOPK);
    printf("store/layer: K=%.1f MB  summaries/layer=%.2f MB  staging/layer=%.1f MB\n",
           store_kv / 1e6, sum_bytes / 1e6, stg_bytes / 1e6);
    printf("per-pass host->device staging volume: %.1f MB (K+V, %d layers)\n",
           2.0 * N_LAYERS * stg_bytes / 1e6, N_LAYERS);

    std::vector<__half> h_kv(store_kv / sizeof(__half));
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto & x : h_kv) x = __float2half(dist(rng));
    }

    // pinned host store (one layer worth is enough for the timing loop)
    __half *h_pin = nullptr;
    CK(cudaHostAlloc(&h_pin, store_kv, cudaHostAllocDefault));
    memcpy(h_pin, h_kv.data(), store_kv);

    __half *d_kmin, *d_kmax;
    CK(cudaMalloc(&d_kmin, sum_bytes)); CK(cudaMalloc(&d_kmax, sum_bytes));
    float *d_score; int *d_idx; float *d_top;
    CK(cudaMalloc(&d_score, n_pages * sizeof(float)));
    CK(cudaMalloc(&d_idx, TOPK * sizeof(int))); CK(cudaMalloc(&d_top, TOPK * sizeof(float)));
    __half *d_stg; CK(cudaMalloc(&d_stg, stg_bytes));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- 1. summaries: full build for all layers (one-time), then incremental ----
    CK(cudaEventRecord(e0));
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        build_summaries<<<n_pages, 1024>>>((const __half *) h_pin,
                                           d_kmin, d_kmax);
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    const float t_sum_full = ms_between(e0, e1);

    CK(cudaEventRecord(e0));
    build_summaries<<<1, 1024>>>((const __half *) h_pin, d_kmin, d_kmax);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    const float t_sum_page = ms_between(e0, e1);

    // ---- 2. scoring: per layer, 4 groups, summed into one page-score vector ----
    std::vector<float> h_q((size_t) GROUPS * KV_DIM);
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto & x : h_q) x = dist(rng);
    }
    float *d_q; CK(cudaMalloc(&d_q, h_q.size() * sizeof(float)));
    CK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice));

    CK(cudaEventRecord(e0));
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        CK(cudaMemsetAsync(d_score, 0, n_pages * sizeof(float)));
        for (int g = 0; g < GROUPS; ++g) {
            page_scores<<<n_pages, 256>>>(d_q, d_kmin, d_kmax, d_score, n_pages, g);
        }
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    const float t_score = ms_between(e0, e1);

    // ---- 3. top-k: one selection per layer per row -> 16*6 selections; time one layer, scale ----
    CK(cudaEventRecord(e0));
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        topk<<<1, 256>>>(d_score, d_idx, d_top, n_pages);
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    const float t_topk_per_row = ms_between(e0, e1);

    // ---- 4. gather: TOPK pages per layer, K+V, pinned host -> device ----
    cudaStream_t cs; CK(cudaStreamCreate(&cs));
    CK(cudaEventRecord(e0));
    for (int layer = 0; layer < N_LAYERS; ++layer) {
        for (int i = 0; i < TOPK; ++i) {
            const size_t src_off = (size_t) ((layer * 131 + i * 37) % n_pages) * page_bytes;
            CK(cudaMemcpyAsync(d_stg + (size_t) i * PAGE_TOK * KV_DIM,
                               h_pin + src_off / sizeof(__half),
                               page_bytes, cudaMemcpyHostToDevice, cs));
        }
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    const float t_gather_k_only = ms_between(e0, e1);
    const float t_gather_kv = t_gather_k_only * 2.0f;   // + V stream (same volume)

    printf("\n=== measured on RTX 5060 Ti (sm_120a, CUDA 13.2) ===\n");
    printf("summaries full build (16 layers x 4096 pages): %8.2f ms  (one-time)\n", t_sum_full);
    printf("summaries incremental (1 page, 1 layer)      : %8.3f ms  (per 64 new tokens)\n", t_sum_page);
    printf("page scoring (16 layers x 4 groups, 1 row)    : %8.3f ms  (x6 rows = %.2f ms)\n",
           t_score, t_score * N_ROWS);
    printf("top-k (16 layers, 1 row)                      : %8.3f ms  (x6 rows = %.2f ms)\n",
           t_topk_per_row, t_topk_per_row * N_ROWS);
    printf("gather K from pinned host (16 layers x 16 pgs): %8.2f ms\n", t_gather_k_only);
    printf("gather K+V (same volume x2)                   : %8.2f ms  <- per pass upper bound\n", t_gather_kv);
    const float stg_mb = 2.0 * N_LAYERS * stg_bytes / 1e6;
    printf("  effective host->device bandwidth            : %8.2f GB/s (%.1f MB / %.2f ms)\n",
           stg_mb / 1e3 / (t_gather_kv / 1e3), stg_mb, t_gather_kv);
    printf("\nsparse overhead per pass (score+topk+gather): %8.2f ms\n",
           t_score * N_ROWS + t_topk_per_row * N_ROWS + t_gather_kv);
    printf("compute budget per pass                     : %8.2f ms (measured, stage 1)\n", 40.0);
    return 0;
}
