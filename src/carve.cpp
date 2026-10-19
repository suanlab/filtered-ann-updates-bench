// carve.cpp — recursive randomized ball carving for RIG (ctypes ABI).
// Build: g++ -O3 -march=native -fopenmp -shared -fPIC -o libcarve.so carve.cpp
//
// Partition semantics (C++ counterpart of experiments/exp0_regen_gap.py::rbc_carve, made
// scalable): a subproblem P with |P| > Cmax picks L = min(Lcap, max(2, ceil(|P| * f / T)))
// leaders — the L points of P with the smallest hash priority — and assigns every point of P
// to its f nearest leaders (ties broken by leader id). f follows the fanout schedule f0, f1,
// then 1. Children of size <= Cmax become leaves (kept if >= min_leaf points, as in the Python
// carver); larger children recurse. At n = 1M with f0=10, f1=3, T=60, Lcap=1000 this yields the
// same leader counts as the Python carver (1000 top leaders, ~500 per bucket); at any n it
// yields leaves of roughly T points. Leaders come from a hash of (seed, id), so the partition
// is a deterministic function of the point set; leaf ids follow subproblem order, so output is
// identical across runs and thread counts.
//
// The leader tree is retained so that new points can be assigned incrementally
// (carve_assign), which is the insert path of a RIG.
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <omp.h>

static inline uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}

static inline float l2(const float* a, const float* b, int d) {
    float s = 0.f;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

struct Node {                       // leader-tree node
    int32_t leaf = -1;              // >= 0: leaf id (terminal); -1: internal
    int depth = 0;
    std::vector<int32_t> leaders;   // internal: leader point ids, ascending by id
    std::vector<int32_t> child;     // internal: child node per leader (-1 if dropped)
};

struct Carve {
    int d = 0; int64_t n = 0;
    std::vector<Node> nodes;                    // nodes[0] = root
    std::vector<std::vector<int32_t>> leaves;   // leaf id -> member point ids (ascending)
    std::vector<int64_t> leaf_off;              // CSR, built on finalize
    std::vector<int32_t> leaf_ids;
    std::vector<int64_t> mem_off;               // point -> leaf ids CSR
    std::vector<int32_t> mem_ids;
};

struct Params { int f0, f1, Cmax, Lcap, T, min_leaf; uint64_t seed; };

static inline int fanout_at(const Params& p, int depth) { return depth == 0 ? p.f0 : (depth == 1 ? p.f1 : 1); }

// Assign each point in P to its f nearest leaders. out[i*f + j] = leader index (into L), sorted
// by (distance, leader id).
static void assign_block(const float* X, int d, const std::vector<int32_t>& P, const std::vector<int32_t>& L,
                         int f, std::vector<int32_t>& out, bool parallel) {
    const size_t np = P.size(); const int nl = (int)L.size();
    out.assign(np * f, -1);
    auto work = [&](size_t i) {
        float bd[16]; int bi[16]; int cnt = 0;
        const float* x = X + (size_t)P[i] * d;
        for (int j = 0; j < nl; ++j) {
            float dd = l2(x, X + (size_t)L[j] * d, d);
            if (cnt < f || dd < bd[cnt - 1] || (dd == bd[cnt - 1] && L[j] < L[bi[cnt - 1]])) {
                int pos = (cnt < f) ? cnt++ : f - 1;
                while (pos > 0 && (dd < bd[pos - 1] || (dd == bd[pos - 1] && L[j] < L[bi[pos - 1]]))) {
                    bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; --pos;
                }
                bd[pos] = dd; bi[pos] = j;
            }
        }
        for (int t = 0; t < cnt; ++t) out[i * f + t] = bi[t];
    };
    if (parallel) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < np; ++i) work(i);
    } else {
        for (size_t i = 0; i < np; ++i) work(i);
    }
}

// Choose leaders of subproblem P: the L smallest-priority points (returned ascending by id).
static std::vector<int32_t> pick_leaders(const std::vector<int32_t>& P, int L, uint64_t seed) {
    std::vector<std::pair<uint64_t, int32_t>> pr(P.size());
    for (size_t i = 0; i < P.size(); ++i) pr[i] = {mix64((uint64_t)P[i] ^ seed), P[i]};
    if ((size_t)L < pr.size()) std::nth_element(pr.begin(), pr.begin() + L, pr.end());
    std::vector<int32_t> out(L);
    for (int i = 0; i < L; ++i) out[i] = pr[i].second;
    std::sort(out.begin(), out.end());
    return out;
}

// Carve one subproblem (serial inside). Produces child point lists; caller turns them into nodes.
static void split(const float* X, int d, const Params& p, const std::vector<int32_t>& P, int depth,
                  std::vector<int32_t>& leaders, std::vector<std::vector<int32_t>>& kids, bool parallel) {
    int f = fanout_at(p, depth);
    long want = (long)std::ceil((double)P.size() * f / p.T);
    int L = (int)std::min<long>(p.Lcap, std::max<long>(2, want));
    L = std::min<int>(L, (int)P.size());
    f = std::min(f, L);
    leaders = pick_leaders(P, L, p.seed);
    std::vector<int32_t> asg;
    assign_block(X, d, P, leaders, f, asg, parallel);
    std::vector<int64_t> cnt(L + 1, 0);
    for (size_t i = 0; i < asg.size(); ++i) if (asg[i] >= 0) cnt[asg[i] + 1]++;
    kids.assign(L, {});
    for (int j = 0; j < L; ++j) kids[j].reserve(cnt[j + 1]);
    for (size_t i = 0; i < P.size(); ++i)          // P is ascending -> kids stay ascending
        for (int t = 0; t < f; ++t) { int j = asg[i * f + t]; if (j >= 0) kids[j].push_back(P[i]); }
}

extern "C" {

// X has n rows; if subset != NULL only those nsub rows are carved (ids stay X row ids, and the
// membership CSR still spans all n rows, empty for rows outside the subset).
void* carve_run(const float* X, int64_t n, int d, const int32_t* subset, int64_t nsub, uint64_t seed, int f0, int f1,
                int Cmax, int Lcap, int T, int min_leaf, int nthreads) {
    omp_set_num_threads(nthreads);
    Params p{f0, f1, Cmax, Lcap, T, min_leaf, seed};
    Carve* C = new Carve(); C->d = d; C->n = n;
    // frontier of (node id, point list)
    std::vector<int32_t> all;
    if (subset) { all.assign(subset, subset + nsub); std::sort(all.begin(), all.end()); }
    else { all.resize(n); for (int64_t i = 0; i < n; ++i) all[i] = (int32_t)i; }
    C->nodes.emplace_back();
    std::vector<std::pair<int32_t, std::vector<int32_t>>> frontier;
    frontier.push_back({0, std::move(all)});
    int depth = 0;
    while (!frontier.empty()) {
        // split every frontier subproblem; large ones get intra-subproblem parallelism
        const size_t F = frontier.size();
        std::vector<std::vector<int32_t>> leadersv(F);
        std::vector<std::vector<std::vector<int32_t>>> kidsv(F);
        std::vector<char> is_leaf(F, 0);
        for (size_t s = 0; s < F; ++s) {
            auto& P = frontier[s].second;
            bool stuck = depth > 12;
            if ((int64_t)P.size() <= Cmax || stuck) { is_leaf[s] = 1; continue; }
        }
        // big subproblems (> 1M) one at a time with parallel assignment; the rest in parallel
        std::vector<size_t> big, small;
        for (size_t s = 0; s < F; ++s) if (!is_leaf[s]) (frontier[s].second.size() > 1000000 ? big : small).push_back(s);
        for (size_t s : big) split(X, d, p, frontier[s].second, depth, leadersv[s], kidsv[s], true);
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t t = 0; t < small.size(); ++t) {
            size_t s = small[t];
            split(X, d, p, frontier[s].second, depth, leadersv[s], kidsv[s], false);
        }
        // materialize nodes/leaves in deterministic frontier order
        std::vector<std::pair<int32_t, std::vector<int32_t>>> next;
        for (size_t s = 0; s < F; ++s) {
            int32_t nid = frontier[s].first;
            auto& P = frontier[s].second;
            if (is_leaf[s]) {
                // terminal: split oversized stuck lists into Cmax chunks (each a leaf)
                if ((int64_t)P.size() <= Cmax) {
                    if ((int)P.size() >= min_leaf) { C->nodes[nid].leaf = (int32_t)C->leaves.size(); C->leaves.push_back(std::move(P)); }
                } else {
                    for (size_t c = 0; c < P.size(); c += Cmax) {
                        std::vector<int32_t> part(P.begin() + c, P.begin() + std::min(P.size(), c + (size_t)Cmax));
                        if ((int)part.size() >= min_leaf) { if (C->nodes[nid].leaf < 0) C->nodes[nid].leaf = (int32_t)C->leaves.size(); C->leaves.push_back(std::move(part)); }
                    }
                }
                continue;
            }
            // No reference into C->nodes may be held across emplace_back (reallocation).
            const size_t NLd = leadersv[s].size();
            C->nodes[nid].depth = depth; C->nodes[nid].leaders = std::move(leadersv[s]);
            C->nodes[nid].child.assign(NLd, -1);
            for (size_t j = 0; j < NLd; ++j) {
                auto& kid = kidsv[s][j];
                if (kid.empty()) continue;
                if ((int64_t)kid.size() <= Cmax && (int)kid.size() < min_leaf) continue;  // dropped (as in Python)
                int32_t cid = (int32_t)C->nodes.size();
                C->nodes.emplace_back();
                C->nodes[nid].child[j] = cid;
                C->nodes[cid].depth = depth + 1;
                next.push_back({cid, std::move(kid)});
            }
        }
        frontier.swap(next);
        ++depth;
    }
    // CSR of leaves
    const size_t NL = C->leaves.size();
    C->leaf_off.assign(NL + 1, 0);
    for (size_t l = 0; l < NL; ++l) C->leaf_off[l + 1] = C->leaf_off[l] + (int64_t)C->leaves[l].size();
    C->leaf_ids.resize(C->leaf_off[NL]);
    #pragma omp parallel for schedule(static)
    for (size_t l = 0; l < NL; ++l) std::copy(C->leaves[l].begin(), C->leaves[l].end(), C->leaf_ids.begin() + C->leaf_off[l]);
    // membership CSR (point -> leaf ids, ascending)
    C->mem_off.assign(n + 1, 0);
    for (size_t s = 0; s < C->leaf_ids.size(); ++s) C->mem_off[(size_t)C->leaf_ids[s] + 1]++;
    for (int64_t i = 0; i < n; ++i) C->mem_off[i + 1] += C->mem_off[i];
    C->mem_ids.resize(C->mem_off[n]);
    std::vector<int64_t> fill(C->mem_off.begin(), C->mem_off.end() - 1);
    for (size_t l = 0; l < NL; ++l)
        for (int64_t s = C->leaf_off[l]; s < C->leaf_off[l + 1]; ++s) C->mem_ids[fill[C->leaf_ids[s]]++] = (int32_t)l;
    for (auto& v : C->leaves) { std::vector<int32_t>().swap(v); }   // CSR is the source of truth now
    return C;
}

void carve_sizes(void* h, int64_t* nleaves, int64_t* nslots, int64_t* nnodes, int64_t* nleaders0) {
    Carve* C = (Carve*)h;
    *nleaves = (int64_t)C->leaf_off.size() - 1; *nslots = (int64_t)C->leaf_ids.size();
    *nnodes = (int64_t)C->nodes.size(); *nleaders0 = (int64_t)C->nodes[0].leaders.size();
}

void carve_copy(void* h, int64_t* leaf_off, int32_t* leaf_ids, int64_t* mem_off, int32_t* mem_ids, int32_t* leaders0) {
    Carve* C = (Carve*)h;
    std::memcpy(leaf_off, C->leaf_off.data(), C->leaf_off.size() * sizeof(int64_t));
    std::memcpy(leaf_ids, C->leaf_ids.data(), C->leaf_ids.size() * sizeof(int32_t));
    std::memcpy(mem_off, C->mem_off.data(), C->mem_off.size() * sizeof(int64_t));
    std::memcpy(mem_ids, C->mem_ids.data(), C->mem_ids.size() * sizeof(int32_t));
    std::memcpy(leaders0, C->nodes[0].leaders.data(), C->nodes[0].leaders.size() * sizeof(int32_t));
}

// Incremental insert path: route each new point down the leader tree with the same fanout
// schedule and report the leaves it lands in. Xall must contain the new points at rows newids.
// Returns the number of distance computations performed (the insert's work).
int64_t carve_assign(void* h, const float* Xall, const int32_t* newids, int64_t nnew, int Cmax_unused,
                     int f0, int f1, int nthreads, int64_t* out_nleaves_per_point) {
    Carve* C = (Carve*)h; const int d = C->d;
    Params p{f0, f1, 0, 0, 0, 0, 0};
    omp_set_num_threads(nthreads);
    int64_t total = 0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total)
    for (int64_t i = 0; i < nnew; ++i) {
        const float* x = Xall + (size_t)newids[i] * d;
        std::vector<int32_t> stack{0}; int64_t landed = 0;
        while (!stack.empty()) {
            int32_t nid = stack.back(); stack.pop_back();
            const Node& nd = C->nodes[nid];
            if (nd.leaf >= 0 || nd.leaders.empty()) { if (nd.leaf >= 0) ++landed; continue; }
            int f = std::min<int>(fanout_at(p, nd.depth), (int)nd.leaders.size());
            float bd[16]; int bi[16]; int cnt = 0;
            for (size_t j = 0; j < nd.leaders.size(); ++j) {
                float dd = l2(x, Xall + (size_t)nd.leaders[j] * d, d); ++total;
                if (cnt < f || dd < bd[cnt - 1]) {
                    int pos = (cnt < f) ? cnt++ : f - 1;
                    while (pos > 0 && dd < bd[pos - 1]) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; --pos; }
                    bd[pos] = dd; bi[pos] = (int)j;
                }
            }
            for (int t = 0; t < cnt; ++t) { int32_t c = nd.child[bi[t]]; if (c >= 0) stack.push_back(c); }
        }
        out_nleaves_per_point[i] = landed;
    }
    return total;
}

// ---------------------------------------------------------------------------------------------
// Dynamic maintenance, "pivot-retaining" path (used by the TCO stream experiment).
//  insert: route each new point down the leader tree (same fanout schedule) and append it to the
//          leaves it lands in. Leaders never change, so this path is NOT state-identical to a
//          fresh carve; the exact protocol (leader re-election) is evaluated separately (exp5b).
//  delete: the point leaves every leaf (lazily, via an alive bitmap applied at export). If it was
//          a leader, its coordinates stay in use as a routing pivot ("ghost pivot").
//  export: rebuilds the live-only CSR of leaves and memberships (a compaction step).
// ---------------------------------------------------------------------------------------------
struct Dyn { std::vector<std::vector<int32_t>> leaves; std::vector<uint8_t> alive; int64_t cap = 0; };
struct CarveDyn { Carve* C; Dyn D; };

void* carve_dyn_open(void* h, const int32_t* alive_ids, int64_t nalive) {
    Carve* C = (Carve*)h; CarveDyn* cd = new CarveDyn(); cd->C = C; const int64_t capacity = C->n;
    const size_t NL = C->leaf_off.size() - 1;
    cd->D.leaves.resize(NL);
    for (size_t l = 0; l < NL; ++l) cd->D.leaves[l].assign(C->leaf_ids.begin() + C->leaf_off[l], C->leaf_ids.begin() + C->leaf_off[l + 1]);
    cd->D.cap = capacity; cd->D.alive.assign(capacity, 0);
    for (int64_t i = 0; i < nalive; ++i) cd->D.alive[alive_ids[i]] = 1;
    return cd;
}

// returns distance computations performed (the insert work)
int64_t carve_dyn_insert(void* hd, const float* Xall, const int32_t* ids, int64_t nnew, int f0, int f1, int nthreads) {
    CarveDyn* cd = (CarveDyn*)hd; Carve* C = cd->C; const int d = C->d;
    Params p{f0, f1, 0, 0, 0, 0, 0};
    omp_set_num_threads(nthreads);
    std::vector<std::vector<std::pair<int32_t, int32_t>>> per_thread(omp_get_max_threads());
    int64_t total = 0;
    #pragma omp parallel reduction(+:total)
    {
        auto& out = per_thread[omp_get_thread_num()];
        #pragma omp for schedule(dynamic, 64)
        for (int64_t i = 0; i < nnew; ++i) {
            const float* x = Xall + (size_t)ids[i] * d;
            std::vector<int32_t> stack{0};
            while (!stack.empty()) {
                int32_t nid = stack.back(); stack.pop_back();
                const Node& nd = C->nodes[nid];
                if (nd.leaf >= 0) { out.push_back({nd.leaf, ids[i]}); continue; }
                if (nd.leaders.empty()) continue;
                int f = std::min<int>(fanout_at(p, nd.depth), (int)nd.leaders.size());
                float bd[16]; int bi[16]; int cnt = 0;
                for (size_t j = 0; j < nd.leaders.size(); ++j) {
                    float dd = l2(x, Xall + (size_t)nd.leaders[j] * d, d); ++total;
                    if (cnt < f || dd < bd[cnt - 1]) {
                        int pos = (cnt < f) ? cnt++ : f - 1;
                        while (pos > 0 && dd < bd[pos - 1]) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; --pos; }
                        bd[pos] = dd; bi[pos] = (int)j;
                    }
                }
                for (int t = 0; t < cnt; ++t) { int32_t c = nd.child[bi[t]]; if (c >= 0) stack.push_back(c); }
            }
        }
    }
    std::vector<std::pair<int32_t, int32_t>> all;
    for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());                 // deterministic append order
    for (auto& pr : all) cd->D.leaves[pr.first].push_back(pr.second);
    for (int64_t i = 0; i < nnew; ++i) cd->D.alive[ids[i]] = 1;
    return total;
}

void carve_dyn_delete(void* hd, const int32_t* ids, int64_t nd_) {
    CarveDyn* cd = (CarveDyn*)hd;
    for (int64_t i = 0; i < nd_; ++i) cd->D.alive[ids[i]] = 0;
}

// Ghost pivots = leaders (at any level) whose point is no longer alive.
int64_t carve_dyn_ghost_pivots(void* hd) {
    CarveDyn* cd = (CarveDyn*)hd; int64_t g = 0;
    for (const Node& nd : cd->C->nodes) for (int32_t l : nd.leaders) if (!cd->D.alive[l]) ++g;
    return g;
}

// Compaction: live-only CSR. Sizes first (out arrays NULL), then fill.
void carve_dyn_export(void* hd, int64_t* nslots_out, int64_t* leaf_off, int32_t* leaf_ids,
                      int64_t* mem_off, int32_t* mem_ids, int nthreads) {
    CarveDyn* cd = (CarveDyn*)hd; const size_t NL = cd->D.leaves.size(); const int64_t cap = cd->D.cap;
    std::vector<int64_t> off(NL + 1, 0);
    for (size_t l = 0; l < NL; ++l) { int64_t c = 0; for (int32_t x : cd->D.leaves[l]) c += cd->D.alive[x]; off[l + 1] = off[l] + c; }
    *nslots_out = off[NL];
    if (!leaf_off) return;
    std::memcpy(leaf_off, off.data(), off.size() * sizeof(int64_t));
    omp_set_num_threads(nthreads);
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t l = 0; l < NL; ++l) { int64_t w = off[l]; for (int32_t x : cd->D.leaves[l]) if (cd->D.alive[x]) leaf_ids[w++] = x; }
    // physically drop dead ids from the dynamic leaves too (the compaction)
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t l = 0; l < NL; ++l) {
        auto& v = cd->D.leaves[l]; size_t w = 0;
        for (size_t r = 0; r < v.size(); ++r) if (cd->D.alive[v[r]]) v[w++] = v[r];
        v.resize(w);
    }
    std::fill(mem_off, mem_off + cap + 1, 0);
    for (int64_t s = 0; s < off[NL]; ++s) mem_off[(size_t)leaf_ids[s] + 1]++;
    for (int64_t i = 0; i < cap; ++i) mem_off[i + 1] += mem_off[i];
    std::vector<int64_t> fill(mem_off, mem_off + cap);
    for (size_t l = 0; l < NL; ++l)
        for (int64_t s = off[l]; s < off[l + 1]; ++s) mem_ids[fill[leaf_ids[s]]++] = (int32_t)l;
}

int64_t carve_dyn_nleaves(void* hd) { return (int64_t)((CarveDyn*)hd)->D.leaves.size(); }
void carve_dyn_close(void* hd) { delete (CarveDyn*)hd; }

void carve_free(void* h) { delete (Carve*)h; }

}
