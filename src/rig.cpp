// rig.cpp — C++ kernels for Regenerable Implicit Graph search (ctypes ABI).
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC -o librig.so rig.cpp
//
// GLOBAL-STATE CONTRACT:
//  * rig_set_quant / rig_set_sketch / rig_set_leafmajor install process-wide pointers
//    consumed by rig_search. The caller owns the arrays and must keep them alive and
//    unchanged for the duration of any rig_search call that uses them.
//  * Setters and rig_search must not run concurrently; concurrent rig_search calls
//    sharing the installed state are also unsupported. The Python harness calls them
//    strictly sequentially.
//  * Per-thread scratch (visited/harvest stamps) is thread_local and persists across
//    calls by design (intentional leak); it is resized upward only. Epoch counters make
//    stale stamp values harmless across datasets of differing n.
//  * mode 5/7 require rig_set_quant; mode 7 additionally requires rig_set_leafmajor;
//    cone pruning (prune_mode 1) requires rig_set_sketch. Guarded below.
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>

static inline float sqdist(const float* a, const float* b, int d) {
    float s = 0.f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

struct Scratch {
    std::vector<int32_t> stamp, hstamp, vstamp; int32_t epoch = 0; int32_t hepoch = 0;
    std::vector<int32_t> pool; std::vector<float> du, dq;
    std::vector<int32_t> idx;
    Scratch(int n) : stamp(n, 0), hstamp(n, 0), vstamp(n, 0) {}
};

// robust prune over candidates (ids, dist-to-u), returns kept ids (<= R)
static void robust_prune(const float* X, int d, int u, std::vector<int32_t>& ids, std::vector<float>& ds,
                         float alpha, int R, std::vector<int32_t>& out, int64_t& ncomp) {
    out.clear();
    std::vector<int32_t> order(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) order[i] = (int32_t)i;
    std::sort(order.begin(), order.end(), [&](int a, int b){ return ds[a] < ds[b] || (ds[a] == ds[b] && ids[a] < ids[b]); });
    float a2 = alpha * alpha;
    for (int oi : order) {
        int c = ids[oi]; float dc = ds[oi]; if (c == u) continue;
        bool ok = true;
        for (int y : out) { float dyc = sqdist(X + (size_t)y * d, X + (size_t)c * d, d); ++ncomp; if (a2 * dyc < dc) { ok = false; break; } }
        if (ok) { out.push_back(c); if ((int)out.size() >= R) break; }
    }
}

// gather pool of u (union of its leaves), excluding u; optionally masked
static void gather_pool(int u, const int64_t* leaf_off, const int32_t* leaf_ids, const int64_t* mem_off,
                        const int32_t* mem_ids, const uint8_t* mask, bool use_mask, Scratch& S) {
    S.pool.clear(); ++S.epoch; if (S.epoch == INT32_MAX) { std::fill(S.stamp.begin(), S.stamp.end(), 0); S.epoch = 1; }
    for (int64_t m = mem_off[u]; m < mem_off[u + 1]; ++m) {
        int32_t lf = mem_ids[m];
        for (int64_t j = leaf_off[lf]; j < leaf_off[lf + 1]; ++j) {
            int32_t p = leaf_ids[j];
            if (p == u || S.stamp[p] == S.epoch) continue;
            if (use_mask && !mask[p]) continue;
            S.stamp[p] = S.epoch; S.pool.push_back(p);
        }
    }
}

// regenerate nav neighbors of u from pool; if q != nullptr also compute dq and harvest masked into (hv)
static const int8_t* g_X8 = nullptr; static const float* g_sx = nullptr; static const float* g_xn = nullptr;
static inline float sqdist_q8(const int8_t* q8, float sq, float qn, int32_t p, int d) {
    const int8_t* x8 = g_X8 + (size_t)p * d; int32_t acc = 0;
    #pragma omp simd reduction(+:acc)
    for (int i = 0; i < d; ++i) acc += (int32_t)q8[i] * (int32_t)x8[i];
    float dot = sq * g_sx[p] * (float)acc;
    return qn + g_xn[p] - 2.f * dot;
}
static const float* g_S = nullptr; static int g_m = 0; static int g_prune_mode = 0;

// cone prune on sketches: bucket by sign(S[c]-S[u]) (m bits), keep nearest per cone, up to R nearest cones
static void cone_prune(int u, const std::vector<int32_t>& ids, const std::vector<float>& ds, int R, std::vector<int32_t>& out) {
    const float* su = g_S + (size_t)u * g_m;
    std::vector<std::pair<float,int32_t>> best; // per-cone nearest; use small open-addressing via sort
    std::vector<std::pair<uint32_t, int32_t>> keyed(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const float* sc = g_S + (size_t)ids[i] * g_m; uint32_t h = 0;
        for (int b = 0; b < g_m; ++b) h |= (uint32_t)(sc[b] > su[b]) << b;
        keyed[i] = {h, (int32_t)i};
    }
    std::sort(keyed.begin(), keyed.end(), [&](const std::pair<uint32_t,int32_t>& a, const std::pair<uint32_t,int32_t>& b){
        return a.first < b.first || (a.first == b.first && ds[a.second] < ds[b.second]); });
    out.clear();
    for (size_t i = 0; i < keyed.size(); ++i) {
        if (i > 0 && keyed[i].first == keyed[i-1].first) continue; // keep first (nearest) per cone
        best.push_back({ds[keyed[i].second], ids[keyed[i].second]});
    }
    std::sort(best.begin(), best.end());
    for (size_t i = 0; i < best.size() && (int)i < R; ++i) out.push_back(best[i].second);
}

static void regen(const float* X, int d, int u, const float* q, const uint8_t* mask, Scratch& S,
                  float alpha, int R, int topC, std::vector<int32_t>& nbrs,
                  std::vector<std::pair<float,int32_t>>* harvest, int64_t& ncomp) {
    size_t P = S.pool.size(); S.du.resize(P); if (q) S.dq.resize(P);
    const float* xu = X + (size_t)u * d;
    for (size_t i = 0; i < P; ++i) {
        const float* xp = X + (size_t)S.pool[i] * d;
        S.du[i] = sqdist(xu, xp, d); ++ncomp;
        if (q) { S.dq[i] = sqdist(q, xp, d); ++ncomp;
        }
    }
    if (q && harvest) {
        for (size_t i = 0; i < P; ++i) { int32_t p = S.pool[i]; if (mask[p] && S.hstamp[p] != S.hepoch) { S.hstamp[p] = S.hepoch; harvest->push_back({S.dq[i], p}); } }
    }
    // topC by du
    S.idx.resize(P); for (size_t i = 0; i < P; ++i) S.idx[i] = (int32_t)i;
    size_t keep = std::min((size_t)topC, P);
    if (keep < P) std::nth_element(S.idx.begin(), S.idx.begin() + keep, S.idx.end(), [&](int a, int b){ return S.du[a] < S.du[b]; });
    std::vector<int32_t> ids(keep); std::vector<float> ds(keep);
    for (size_t i = 0; i < keep; ++i) { ids[i] = S.pool[S.idx[i]]; ds[i] = S.du[S.idx[i]]; }
    if (g_prune_mode == 1) cone_prune(u, ids, ds, R, nbrs); else robust_prune(X, d, u, ids, ds, alpha, R, nbrs, ncomp);
}

static void entries(const float* X, int d, const float* q, const int32_t* leaders, int L0, const uint8_t* mask,
                    bool use_mask, int t, std::vector<int32_t>& ent, int64_t& ncomp) {
    std::vector<std::pair<float,int32_t>> v; v.reserve(L0);
    for (int i = 0; i < L0; ++i) { int32_t l = leaders[i]; if (use_mask && !mask[l]) continue; v.push_back({sqdist(q, X + (size_t)l * d, d), l}); ++ncomp; }
    std::sort(v.begin(), v.end()); ent.clear();
    for (int i = 0; i < (int)v.size() && i < t; ++i) ent.push_back(v[i].second);
}

static const int8_t* g_LX8 = nullptr; static const float* g_Lsx = nullptr; static const float* g_Lxn = nullptr;
static const int32_t* g_Lid = nullptr; static const int64_t* g_loff8 = nullptr;
extern "C" {
void rig_set_leafmajor(const int8_t* LX8, const float* Lsx, const float* Lxn, const int32_t* Lid, const int64_t* loff8) {
    g_LX8 = LX8; g_Lsx = Lsx; g_Lxn = Lxn; g_Lid = Lid; g_loff8 = loff8; }
void rig_set_quant(const int8_t* X8, const float* sx, const float* xn) { g_X8 = X8; g_sx = sx; g_xn = xn; }
void rig_set_sketch(const float* S, int m, int prune_mode) { g_S = S; g_m = m; g_prune_mode = prune_mode; }


// mode: 8 = int8-quantized distances over the leaf-major layout with navigation restricted to admissible
// candidates -- the only mode behind RIG numbers in the benchmark (rigx.RigIndex defaults: leafmajor, filtered_nav).
// LEGACY, kept for the earlier RIG experiments (experiments/) and not used by the benchmark: 0 = explicit CSR graph
// (in-filter if use_mask), 1 = RIG masked pool, 2 = RIG route-harvest, 3 = RIG unmasked, 4 = exact-distance pool,
// 5 = mode 4 with int8-quantized distances, 7 = mode 5 over the leaf-major layout (unfiltered navigation).
#include <cstdio>
void rig_search64(const float* X, int n, int d,
                const int64_t* leaf_off, const int32_t* leaf_ids, const int64_t* mem_off, const int32_t* mem_ids,
                const int64_t* g_off, const int32_t* g_ids,
                const int32_t* leaders, int L0, const uint8_t* mask, int use_mask,
                const float* Q, int nq, float alpha, int R, int topC, int ef, int k, int t_entry, int mode,
                int nthreads, int32_t* out_ids, int64_t* out_ncomp) {
    if (((mode == 5 || mode == 7 || mode == 8) && !g_X8) || ((mode == 7 || mode == 8) && !g_LX8) || (g_prune_mode == 1 && !g_S)) {
        std::fprintf(stderr, "rig_search: required global state not installed for mode %d\n", mode);
        for (long long i = 0; i < (long long)nq * k; ++i) out_ids[i] = -1;
        for (int i = 0; i < nq; ++i) out_ncomp[i] = 0;
        return;
    }
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {
        static thread_local Scratch* S_tl = nullptr;
        if (!S_tl || (int)S_tl->stamp.size() < n) { delete S_tl; S_tl = new Scratch(n); }
        Scratch& S = *S_tl;
        std::vector<int32_t> nbrs, ent;
        std::vector<std::pair<float,int32_t>> harvest;
        #pragma omp for schedule(dynamic, 4)
        for (int qi = 0; qi < nq; ++qi) {
            const float* q = Q + (size_t)qi * d; int64_t ncomp = 0;
            std::vector<int8_t> q8; float sq = 1.f, qn = 0.f;
            if (mode == 5 || mode == 7 || mode == 8) { q8.resize(d); float mx = 0.f; for (int i = 0; i < d; ++i) { mx = std::max(mx, std::fabs(q[i])); qn += q[i] * q[i]; }
                sq = (mx > 0.f) ? mx / 127.f : 1.f; for (int i = 0; i < d; ++i) q8[i] = (int8_t)std::lrint(q[i] / sq); }
            bool nav_masked = use_mask && (mode == 0 || mode == 1 || mode == 8);
            entries(X, d, q, leaders, L0, mask, nav_masked, t_entry, ent, ncomp);
            // visited via vstamp with per-query epoch
            // visited-stamp epoch: a per-thread counter (thread-local scratch), reset before it overflows
            int32_t vep; static thread_local int32_t vcount = 0; ++vcount; if (vcount == INT32_MAX) { std::fill(S.vstamp.begin(), S.vstamp.end(), 0); vcount = 1; }
            vep = vcount; ++S.hepoch; if (S.hepoch == INT32_MAX) { std::fill(S.hstamp.begin(), S.hstamp.end(), 0); S.hepoch = 1; }
            harvest.clear();
            typedef std::pair<float,int32_t> PI;
            std::priority_queue<PI, std::vector<PI>, std::greater<PI>> cand;
            std::priority_queue<PI> top;
            for (int e : ent) { float de = sqdist(q, X + (size_t)e * d, d); ++ncomp; cand.push({de, e}); top.push({de, e}); S.vstamp[e] = vep; }
            while (!cand.empty()) {
                PI cur = cand.top(); cand.pop();
                if ((int)top.size() >= ef && cur.first > top.top().first) break;
                int u = cur.second;
                if (mode == 0) {
                    nbrs.clear(); for (int64_t j = g_off[u]; j < g_off[u + 1]; ++j) { int32_t v = g_ids[j]; if (use_mask && !mask[v]) continue; nbrs.push_back(v); }
                } else if (mode == 4 || mode == 5) {
                    gather_pool(u, leaf_off, leaf_ids, mem_off, mem_ids, mask, false, S);
                    size_t Pn = S.pool.size(); S.dq.resize(Pn); S.idx.resize(Pn);
                    for (size_t i = 0; i < Pn; ++i) { int32_t p = S.pool[i];
                        S.dq[i] = (mode == 5) ? sqdist_q8(q8.data(), sq, qn, p, d) : sqdist(q, X + (size_t)p * d, d); ++ncomp; S.idx[i] = (int32_t)i;
                        if (use_mask && mask[p] && S.hstamp[p] != S.hepoch) { S.hstamp[p] = S.hepoch; harvest.push_back({S.dq[i], p}); } }
                    size_t keep = std::min((size_t)R, Pn);
                    if (keep < Pn) std::nth_element(S.idx.begin(), S.idx.begin() + keep, S.idx.end(), [&](int a, int b){ return S.dq[a] < S.dq[b]; });
                    nbrs.clear(); for (size_t i = 0; i < keep; ++i) nbrs.push_back(S.pool[S.idx[i]]);
                    // push with known dq (skip recompute)
                    for (size_t i = 0; i < keep; ++i) { int v = S.pool[S.idx[i]]; float dv = S.dq[S.idx[i]];
                        if (S.vstamp[v] == vep) continue; S.vstamp[v] = vep;
                        if ((int)top.size() < ef || dv < top.top().first) { cand.push({dv, v}); top.push({dv, v}); if ((int)top.size() > ef) top.pop(); } }
                    nbrs.clear();
                } else if (mode == 7 || mode == 8) {
                    // mode 8 = mode 7 with navigation restricted to predicate-passing candidates
                    // (filtered navigation: needed when the admissible set lies away from the query).
                    // leaf-major sequential scan of u's leaves. Slots are scored as streamed
                    // (duplicates across u's leaves are re-scored: sequential > branchy), but
                    // candidates are DEDUPED before top-R selection — without this, at high
                    // leaf overlap the top-R multiset collapses to a few distinct points and
                    // the beam starves (caught by tests/test_units.py at n=5K).
                    S.pool.clear(); S.dq.clear();
                    ++S.epoch; if (S.epoch == INT32_MAX) { std::fill(S.stamp.begin(), S.stamp.end(), 0); S.epoch = 1; }
                    for (int64_t mi = mem_off[u]; mi < mem_off[u + 1]; ++mi) {
                        int32_t lf = mem_ids[mi];
                        for (int64_t sl = g_loff8[lf]; sl < g_loff8[lf + 1]; ++sl) {
                            // filtered navigation: evaluate the predicate before the distance (as ACORN
                            // does) -- non-admissible slots are never used in mode 8.
                            if (mode == 8 && !mask[g_Lid[sl]]) continue;
                            const int8_t* x8 = g_LX8 + (size_t)sl * d; int32_t acc = 0;
                            #pragma omp simd reduction(+:acc)
                            for (int i = 0; i < d; ++i) acc += (int32_t)q8[i] * (int32_t)x8[i];
                            float dd = qn + g_Lxn[sl] - 2.f * sq * g_Lsx[sl] * (float)acc; ++ncomp;
                            int32_t pid = g_Lid[sl]; if (pid == u) continue;
                            if (use_mask && mask[pid] && S.hstamp[pid] != S.hepoch) { S.hstamp[pid] = S.hepoch; harvest.push_back({dd, pid}); }
                            if (mode == 8 && !mask[pid]) continue;    // filtered navigation
                            if (S.stamp[pid] == S.epoch) continue;   // dedupe candidates per hop
                            S.stamp[pid] = S.epoch;
                            S.pool.push_back(pid); S.dq.push_back(dd);
                        }
                    }
                    size_t Pn = S.pool.size(); S.idx.resize(Pn);
                    for (size_t i = 0; i < Pn; ++i) S.idx[i] = (int32_t)i;
                    size_t keep = std::min((size_t)R, Pn);
                    if (keep < Pn) std::nth_element(S.idx.begin(), S.idx.begin() + keep, S.idx.end(), [&](int a, int b){ return S.dq[a] < S.dq[b]; });
                    for (size_t i = 0; i < keep; ++i) { int v = S.pool[S.idx[i]]; float dv = S.dq[S.idx[i]];
                        if (S.vstamp[v] == vep) continue; S.vstamp[v] = vep;
                        if ((int)top.size() < ef || dv < top.top().first) { cand.push({dv, v}); top.push({dv, v}); if ((int)top.size() > ef) top.pop(); } }
                    nbrs.clear();
                } else {
                    gather_pool(u, leaf_off, leaf_ids, mem_off, mem_ids, mask, mode == 1 && use_mask, S);
                    regen(X, d, u, (mode == 2) ? q : nullptr, mask, S, alpha, R, topC, nbrs, (mode == 2) ? &harvest : nullptr, ncomp);
                }
                for (int v : nbrs) {
                    if (S.vstamp[v] == vep) continue; S.vstamp[v] = vep;
                    float dv = sqdist(q, X + (size_t)v * d, d); ++ncomp;
                    if ((int)top.size() < ef || dv < top.top().first) { cand.push({dv, v}); top.push({dv, v}); if ((int)top.size() > ef) top.pop(); }
                }
            }
            // results
            std::vector<PI> res;
            if (mode == 2 || ((mode == 4 || mode == 5 || mode == 7 || mode == 8) && use_mask)) { std::sort(harvest.begin(), harvest.end()); res = harvest; }
            else { while (!top.empty()) { res.push_back(top.top()); top.pop(); } std::sort(res.begin(), res.end()); }
            if (mode == 5 || mode == 7 || mode == 8) { size_t lim = std::min(res.size(), (size_t)(4 * k)); res.resize(lim);
                for (auto& pr : res) { pr.first = sqdist(q, X + (size_t)pr.second * d, d); ++ncomp; } std::sort(res.begin(), res.end()); }
            int w = 0;
            for (auto& pr : res) { if (use_mask && !mask[pr.second]) continue; out_ids[(size_t)qi * k + w] = pr.second; if (++w >= k) break; }
            for (; w < k; ++w) out_ids[(size_t)qi * k + w] = -1;
            out_ncomp[qi] = ncomp;
        }
    }
}

// brute-force filtered baseline (for reference), parallel over queries
void brute_masked(const float* X, int n, int d, const uint8_t* mask, const float* Q, int nq, int k, int nthreads, int32_t* out_ids) {
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = Q + (size_t)qi * d;
        std::priority_queue<std::pair<float,int32_t>> top;
        for (int i = 0; i < n; ++i) { if (!mask[i]) continue; float dd = sqdist(q, X + (size_t)i * d, d);
            if ((int)top.size() < k || dd < top.top().first) { top.push({dd, i}); if ((int)top.size() > k) top.pop(); } }
        std::vector<std::pair<float,int32_t>> res; while (!top.empty()) { res.push_back(top.top()); top.pop(); }
        std::sort(res.begin(), res.end());
        for (int w = 0; w < k; ++w) out_ids[(size_t)qi * k + w] = (w < (int)res.size()) ? res[w].second : -1;
    }
}
}
