// acorn_stream.cpp — ACORN-gamma under an insert/delete stream with predicates (exp22).
// ACORN's own dynamic path: inserts via IndexACORN::add (incremental, on top of ntotal);
// deletes are applied by excluding the point from every query's filter map (alive AND predicate),
// i.e. tombstones that stay in the graph as routing nodes. Faiss ids equal Xall row ids because
// rows are added in increasing contiguous order (initial rows 0..N0-1, then each round's block).
//
// Usage: acorn_stream Xall.fbin Q.fbin attr.i32 qbin.i32 sched_dir N0 R M gamma M_beta
//                     build_threads search_threads efs_csv out_dir
// Optional (environment):
//   PERC=1    at measured rounds, report per label the live admissible subgraph that level-0 search can use:
//             each live node's effective admissible out-neighbors, collected in the order ACORN's search does
//             (direct neighbors, and neighbors of neighbors for list positions >= M_beta, up to 2M found), their
//             mean count, the fraction of nodes with none, and the share of the largest weakly connected component.
//             Computed after the timed searches; not part of maintenance.
//   REPAIR=1  after each round's updates, rewrite the level-0 list of every live node that points at a deleted
//             node: candidates are its live neighbors and the live entries among the first M_beta neighbors of each
//             deleted neighbor; the nearest fill the list (a delete consolidation in the style of FreshDiskANN,
//             without pruning). Upper levels are left as they are. Timed and counted as maintenance (t_repair).
//   REPAIR=2  the same consolidation on every level of a node (upper-level lists use the deleted neighbors' lists
//             on the same level), so that routing from the entry point avoids deleted nodes as well.
//   REPAIR=3  the REPAIR=2 consolidation on the upper levels only (level 0 untouched): isolates the entry routes.
//   REPAIR=4  every level, but slot filling instead of re-ranking: a node's live neighbors keep their place and
//             order, and only the slots of deleted neighbors are refilled, with the nearest live entries among
//             the first M_beta neighbors of those deleted neighbors that are not already listed (REPAIR=1/2
//             re-rank all candidates by distance, which can push live long-range neighbors out of the list).
//   MECH=1    mechanism instrumentation (follow-up 2, I17), untimed, at measured rounds (implies PERC):
//             m1 effective admissible level-0 degree; m2 length of live level-0 edges (absolute, and normalized by
//             the node's exact 10th-nearest live neighbor on a node sample), with the share above the round-0
//             90th percentile; m3 (in the repair event) per label, the share of a repaired node's live level-0
//             neighbors, and of its long ones, that survive the repair; m4 per query, ACORN's filtered greedy
//             descent of the upper levels replayed exactly (gamma > 1: admissible direct neighbors, up to M) and,
//             from the level-0 entry it reaches, the share of the label's live nodes and of the query's true top-10
//             reachable through effective admissible edges; m5 ACORN's search counters per query at two efSearch
//             values. Needs GT_DIR/gt_r<round>.i32 (global ids). FRESH=1 also builds a new index on the final
//             round's live points and reports its sweep and the same metrics (round R, "fresh":1).
//   ATTR_CSR=<dir>  multi-label points and AND predicates (follow-up 2, I18b): <dir>/ptag_ptr.i64, ptag_idx.i32 (each
//             point's sorted tags), qtag_ptr.i64, qtag_idx.i32 (each query's tags); a point is admissible for a query if it
//             is live and carries all of the query's tags. attr/qbin are then unused (PERC, MECH, E2E not supported).
//   E2E=1     end-to-end timing (follow-up 2, I19): one filter map per label kept current by the updates (timed as
//             maintenance, t_pred) instead of a map per query built outside the timers; each query is its own
//             search call on its label's map, timed together, with per-query latency percentiles.
#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include <numeric>
#include <functional>
#include <cmath>

static std::vector<float> read_fbin(const char* p, int& n, int& d) {
    FILE* f = std::fopen(p, "rb"); if (!f) { std::perror(p); std::exit(1); }
    int32_t h[2]; if (std::fread(h, 4, 2, f) != 2) std::exit(1); n = h[0]; d = h[1];
    std::vector<float> v((size_t)n * d); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1);
    std::fclose(f); return v;
}
static std::vector<int32_t> read_i32(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb"); if (!f) { std::perror(p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1);
    std::fclose(f); return v;
}
// current resident memory (MB): anonymous (heap: data copy and index) and total, from /proc/self/status
static void rss_mb(double& anon, double& total) {
    FILE* f = std::fopen("/proc/self/status", "r"); char line[256]; if (!f) return;
    while (std::fgets(line, sizeof line, f)) {
        if (!std::strncmp(line, "RssAnon:", 8)) anon = std::atof(line + 8) / 1024.0;
        else if (!std::strncmp(line, "VmRSS:", 6)) total = std::atof(line + 6) / 1024.0;
    }
    std::fclose(f);
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- MECH=1 instrumentation (untimed)
struct MechThr { double abs = -1, norm = -1; };
static inline float l2(const float* a, const float* b, int d) { float s = 0; for (int t = 0; t < d; ++t) { float df = a[t] - b[t]; s += df * df; } return s; }
// effective admissible level-0 out-neighbors of u for label lab, in the order ACORN's level-0 search collects them
static void effective(const faiss::ACORN& g, long u, int lab, const std::vector<uint8_t>& alive, const std::vector<int32_t>& attr,
                      int M, int gamma, int Mb, std::vector<int>& out_u) {
    const int cap = 2 * M; out_u.clear(); size_t b, e; g.neighbor_range(u, 0, &b, &e);
    for (size_t j = b; j < e && (int)out_u.size() < cap; ++j) {
        int v = g.neighbors[j]; if (v < 0) break;
        if (alive[v] && attr[v] == lab) out_u.push_back(v);
        if ((int)(j - b) >= Mb || gamma == 1) {
            size_t b2, e2; g.neighbor_range(v, 0, &b2, &e2);
            for (size_t j2 = b2; j2 < e2 && (int)out_u.size() < cap; ++j2) { int w = g.neighbors[j2]; if (w < 0) break; if (w != u && alive[w] && attr[w] == lab) out_u.push_back(w); }
        }
    }
    std::sort(out_u.begin(), out_u.end()); out_u.erase(std::unique(out_u.begin(), out_u.end()), out_u.end());
}
// gid: local id -> global id (identity for the stream index); gt: nq x 10 global ids
static void mech_report(faiss::IndexACORNFlat& index, int r, bool fresh, const float* X, int d, const std::vector<uint8_t>& alive,
                        const std::vector<int32_t>& attr, const std::vector<long>& gid, const float* Q, int nq, const std::vector<int32_t>& qbin,
                        const std::vector<int32_t>& gt, int M, int gamma, int Mb, MechThr& thr, int bth, const std::vector<char>& fmap) {
    auto& g = index.acorn; const long nt = (long)index.ntotal; int L = 0;
    for (long u = 0; u < nt; ++u) if (alive[u]) L = std::max(L, attr[u] + 1);
    for (int q = 0; q < nq; ++q) L = std::max(L, qbin[q] + 1);
    omp_set_num_threads(bth);
    // m2: absolute lengths of live direct level-0 edges
    std::vector<double> slen(L, 0), cnt(L, 0), clong(L, 0);
    if (thr.abs < 0) {                       // round-0 threshold from every 16th live node
        std::vector<float> ls;
        for (long u = 0; u < nt; u += 16) { if (!alive[u]) continue; size_t b, e; g.neighbor_range(u, 0, &b, &e);
            for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (alive[v]) ls.push_back(std::sqrt(l2(X + (size_t)u * d, X + (size_t)v * d, d))); } }
        std::nth_element(ls.begin(), ls.begin() + ls.size() * 9 / 10, ls.end()); thr.abs = ls[ls.size() * 9 / 10];
    }
    #pragma omp parallel
    {
        std::vector<double> a(L, 0), c(L, 0), cl(L, 0);
        #pragma omp for schedule(dynamic, 1024) nowait
        for (long u = 0; u < nt; ++u) {
            if (!alive[u]) continue; size_t b, e; g.neighbor_range(u, 0, &b, &e); const int l = attr[u];
            for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (!alive[v]) continue;
                double x = std::sqrt(l2(X + (size_t)u * d, X + (size_t)v * d, d)); a[l] += x; c[l] += 1; cl[l] += x > thr.abs; }
        }
        #pragma omp critical
        for (int l = 0; l < L; ++l) { slen[l] += a[l]; cnt[l] += c[l]; clong[l] += cl[l]; }
    }
    // m2 normalized: every k-th live node, exact 10th live NN
    std::vector<long> live_ids; for (long u = 0; u < nt; ++u) if (alive[u]) live_ids.push_back(u);
    const long S = std::min<long>(20000, (long)live_ids.size()); const long step = std::max<long>(1, (long)live_ids.size() / S);
    std::vector<long> samp; for (long i = 0; i < (long)live_ids.size() && (long)samp.size() < S; i += step) samp.push_back(live_ids[i]);
    std::vector<double> nsum(L, 0), ncnt(L, 0), nlong(L, 0);
    {
        faiss::IndexFlatL2 flat(d); std::vector<float> buf((size_t)live_ids.size() * d);
        for (size_t i = 0; i < live_ids.size(); ++i) std::memcpy(buf.data() + i * d, X + (size_t)live_ids[i] * d, d * sizeof(float));
        flat.add((faiss::idx_t)live_ids.size(), buf.data()); std::vector<float>().swap(buf);
        std::vector<float> qs((size_t)samp.size() * d); for (size_t i = 0; i < samp.size(); ++i) std::memcpy(qs.data() + i * d, X + (size_t)samp[i] * d, d * sizeof(float));
        std::vector<float> Dn((size_t)samp.size() * 11); std::vector<faiss::idx_t> In((size_t)samp.size() * 11);
        flat.search((faiss::idx_t)samp.size(), qs.data(), 11, Dn.data(), In.data());
        std::vector<float> nls;
        for (size_t i = 0; i < samp.size(); ++i) {
            const long u = samp[i]; const double r10 = std::sqrt(std::max(1e-12f, Dn[i * 11 + 10])); size_t b, e; g.neighbor_range(u, 0, &b, &e);
            for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (!alive[v]) continue;
                double x = std::sqrt(l2(X + (size_t)u * d, X + (size_t)v * d, d)) / r10; nls.push_back((float)x); nsum[attr[u]] += x; ncnt[attr[u]] += 1; }
        }
        if (thr.norm < 0 && !nls.empty()) { std::vector<float> t = nls; std::nth_element(t.begin(), t.begin() + t.size() * 9 / 10, t.end()); thr.norm = t[t.size() * 9 / 10]; }
        size_t k = 0;
        for (size_t i = 0; i < samp.size(); ++i) { const long u = samp[i]; size_t b, e; g.neighbor_range(u, 0, &b, &e);
            for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (!alive[v]) continue; nlong[attr[u]] += nls[k++] > thr.norm; } }
    }
    // m1 + m4: effective lists, exact filtered upper-level descent, BFS reachability
    std::vector<long> lab_live(L, 0); for (long u : live_ids) lab_live[attr[u]]++;
    std::vector<long> g2l; { long mx = 0; for (long x : gid) mx = std::max(mx, x); g2l.assign(mx + 1, -1); for (long i = 0; i < (long)gid.size(); ++i) g2l[gid[i]] = i; }
    std::vector<double> reach(L, 0), gtr(L, 0), gtd(L, 0), eok(L, 0), nql(L, 0), degs(L, 0), degn(L, 0);
    std::vector<std::vector<int>> eff(nt);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (long u = 0; u < nt; ++u) if (alive[u]) effective(g, u, attr[u], alive, attr, M, gamma, Mb, eff[u]);
    for (long u = 0; u < nt; ++u) if (alive[u]) { degs[attr[u]] += eff[u].size(); degn[attr[u]] += 1; }
    #pragma omp parallel
    {
        std::vector<int> tmp; std::vector<int> depth(nt, -1); std::vector<long> touched, fr, nx;
        std::vector<double> rr(L, 0), gg(L, 0), gd(L, 0), eo(L, 0), nn(L, 0);
        #pragma omp for schedule(dynamic, 4) nowait
        for (int q = 0; q < nq; ++q) {
            const int lab = qbin[q]; const float* xq = Q + (size_t)q * d; auto pass = [&](long v) { return alive[v] && attr[v] == lab; };
            long nearest = g.entry_point; float dn = l2(xq, X + (size_t)nearest * d, d);
            for (int level = g.max_level; level >= 1; --level) {
                for (;;) { long prev = nearest; int found = 0; size_t b, e; g.neighbor_range(nearest, level, &b, &e);
                    for (size_t i = b; i < e; ++i) { int v = g.neighbors[i]; if (v < 0) break; if (!pass(v)) { if (gamma > 1) continue; }
                        if (pass(v)) { ++found; float dv = l2(xq, X + (size_t)v * d, d); if (dv < dn || !pass(nearest)) { nearest = v; dn = dv; } if (found >= M) break; } }
                    if (nearest == prev) break; }
            }
            eo[lab] += pass(nearest); nn[lab] += 1;
            fr.clear(); touched.clear();
            if (pass(nearest)) { depth[nearest] = 0; touched.push_back(nearest); fr.push_back(nearest); }
            else { effective(g, nearest, lab, alive, attr, M, gamma, Mb, tmp); for (int v : tmp) if (depth[v] < 0) { depth[v] = 1; touched.push_back(v); fr.push_back(v); } }
            while (!fr.empty()) { nx.clear(); for (long u : fr) {
                    for (int v : eff[u]) if (depth[v] < 0) { depth[v] = depth[u] + 1; touched.push_back(v); nx.push_back(v); } } fr.swap(nx); }
            rr[lab] += lab_live[lab] ? (double)touched.size() / lab_live[lab] : 0;
            int ng = 0, nr = 0; double sd = 0;
            for (int j = 0; j < 10; ++j) { int gg_ = gt[(size_t)q * 10 + j]; if (gg_ < 0 || gg_ >= (int)g2l.size() || g2l[gg_] < 0) continue; long lv = g2l[gg_]; ++ng;
                if (depth[lv] >= 0) { ++nr; sd += depth[lv]; } }
            gg[lab] += ng ? (double)nr / ng : 1; gd[lab] += nr ? sd / nr : 0;
            for (long v : touched) depth[v] = -1;
        }
        #pragma omp critical
        for (int l = 0; l < L; ++l) { reach[l] += rr[l]; gtr[l] += gg[l]; gtd[l] += gd[l]; eok[l] += eo[l]; nql[l] += nn[l]; }
    }
    std::vector<std::vector<int>>().swap(eff);
    // m5: ACORN's counters at two efSearch values (single thread, untimed)
    std::string m5; const int k = 10; std::vector<float> Dk((size_t)nq * k); std::vector<faiss::idx_t> Ik((size_t)nq * k);
    omp_set_num_threads(1);
    for (int ef : {64, 256}) { index.acorn.efSearch = ef; faiss::acorn_stats.reset(); index.search(nq, Q, k, Dk.data(), Ik.data(), const_cast<char*>(fmap.data()));
        auto& st = faiss::acorn_stats; m5 += std::string(m5.empty() ? "" : ",") + "{\"ef\":" + std::to_string(ef) + ",\"n1\":" + std::to_string(st.n1 / (double)nq) + ",\"n2\":" + std::to_string(st.n2 / (double)nq)
                  + ",\"n3\":" + std::to_string(st.n3 / (double)nq) + ",\"ndis\":" + std::to_string(st.ndis / (double)nq) + "}"; }
    omp_set_num_threads(bth);
    std::printf("{\"event\":\"mech\",\"round\":%d,\"fresh\":%d,\"thr_abs\":%.5f,\"thr_norm\":%.5f,\"m5\":[%s],\"labels\":[", r, (int)fresh, thr.abs, thr.norm, m5.c_str());
    bool first = true;
    for (int l = 0; l < L; ++l) {
        if (!lab_live[l] && !nql[l]) continue;
        std::printf("%s{\"label\":%d,\"live\":%ld,\"deg_mean\":%.3f,\"len_mean\":%.5f,\"long_share\":%.5f,\"nlen_mean\":%.5f,\"nlong_share\":%.5f,\"nq\":%.0f,\"entry_ok\":%.4f,\"reach_frac\":%.5f,\"gt_reach\":%.5f,\"gt_depth\":%.3f}",
            first ? "" : ",", l, lab_live[l], degn[l] ? degs[l] / degn[l] : 0.0, cnt[l] ? slen[l] / cnt[l] : 0.0, cnt[l] ? clong[l] / cnt[l] : 0.0,
            ncnt[l] ? nsum[l] / ncnt[l] : 0.0, ncnt[l] ? nlong[l] / ncnt[l] : 0.0, nql[l], nql[l] ? eok[l] / nql[l] : 0.0, nql[l] ? reach[l] / nql[l] : 0.0,
            nql[l] ? gtr[l] / nql[l] : 0.0, nql[l] ? gtd[l] / nql[l] : 0.0);
        first = false;
    }
    std::printf("]}\n"); std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 15) { std::fprintf(stderr, "usage: see header\n"); return 1; }
    int N, d, nq, dq;
    std::vector<float> X = read_fbin(argv[1], N, d), Q = read_fbin(argv[2], nq, dq);
    std::vector<int32_t> attr = read_i32(argv[3]), qbin = read_i32(argv[4]);
    std::string sched = argv[5]; int N0 = std::atoi(argv[6]), R = std::atoi(argv[7]);
    int M = std::atoi(argv[8]), gamma = std::atoi(argv[9]), Mb = std::atoi(argv[10]);
    int bth = std::atoi(argv[11]), sth = std::atoi(argv[12]);
    std::vector<int> efs; { std::stringstream ss(argv[13]); std::string t; while (std::getline(ss, t, ',')) efs.push_back(std::atoi(t.c_str())); }
    std::string out = argv[14];
    const char* re = std::getenv("ACORN_REPS"); const int reps = re ? std::max(1, std::atoi(re)) : 1;
    std::vector<int> mrounds; if (const char* m = std::getenv("MEASURE")) { std::stringstream ms(m); std::string t; while (std::getline(ms, t, ',')) mrounds.push_back(std::atoi(t.c_str())); }
    auto measure_round = [&](int r) { return mrounds.empty() || std::find(mrounds.begin(), mrounds.end(), r) != mrounds.end(); };
    const bool mech = std::getenv("MECH") && std::atoi(std::getenv("MECH")) > 0;
    const bool fresh = mech && std::getenv("FRESH") && std::atoi(std::getenv("FRESH")) > 0;
    const bool perc = mech || (std::getenv("PERC") && std::atoi(std::getenv("PERC")) > 0);
    const std::string gtdir = std::getenv("GT_DIR") ? std::getenv("GT_DIR") : "";
    MechThr thr; int Lm = 0; for (int32_t a : attr) Lm = std::max(Lm, a + 1);
    std::vector<long> gid_id(N); std::iota(gid_id.begin(), gid_id.end(), 0L);
    const std::string csr = std::getenv("ATTR_CSR") ? std::getenv("ATTR_CSR") : "";
    std::vector<int64_t> pptr, qptr; std::vector<int32_t> pidx, qidx;
    auto read_i64 = [](const std::string& p) { std::vector<int32_t> v = read_i32(p); std::vector<int64_t> o(v.size() / 2); std::memcpy(o.data(), v.data(), o.size() * 8); return o; };
    if (!csr.empty()) { pptr = read_i64(csr + "/ptag_ptr.i64"); pidx = read_i32(csr + "/ptag_idx.i32"); qptr = read_i64(csr + "/qtag_ptr.i64"); qidx = read_i32(csr + "/qtag_idx.i32");
        if (mech || perc) { std::fprintf(stderr, "ATTR_CSR does not support PERC/MECH\n"); return 3; } }
    auto adm = [&](int q, size_t x) {
        if (csr.empty()) return attr[x] == qbin[q];
        const int32_t* b = pidx.data() + pptr[x]; const int32_t* e = pidx.data() + pptr[x + 1];
        for (int64_t j = qptr[q]; j < qptr[q + 1]; ++j) if (!std::binary_search(b, e, qidx[j])) return false;
        return true; };
    const bool e2e = std::getenv("E2E") && std::atoi(std::getenv("E2E")) > 0 && !mech && csr.empty();
    std::vector<std::vector<char>> lmap;
    const int repair_mode = std::getenv("REPAIR") ? std::atoi(std::getenv("REPAIR")) : 0; const bool repair = repair_mode > 0;
    std::vector<int> metadata(N, 0);                 // sized to full capacity: never indexed out of range
    std::vector<uint8_t> alive(N, 0);
    omp_set_num_threads(bth);
    double t0 = now();
    faiss::IndexACORNFlat index(d, M, gamma, metadata, Mb);
    index.add(N0, X.data());
    for (int i = 0; i < N0; ++i) alive[i] = 1;
    if (e2e) { lmap.assign(Lm, std::vector<char>(N, 0)); for (int i = 0; i < N0; ++i) lmap[attr[i]][i] = 1; }
    std::printf("{\"event\":\"build\",\"n\":%d,\"build_s\":%.2f}\n", N0, now() - t0); std::fflush(stdout);
    const int k = 10;
    std::vector<float> D((size_t)nq * k); std::vector<faiss::idx_t> I((size_t)nq * k);
    double cum = 0;
    for (int r = 0; r <= R; ++r) {
        double t_del = 0, t_ins = 0, t_pred = 0;
        if (r > 0) {
            std::vector<int32_t> del = read_i32(sched + "/del" + std::to_string(r - 1) + ".i32");
            std::vector<int32_t> ins = read_i32(sched + "/ins" + std::to_string(r - 1) + ".i32");
            double a = now(); for (int32_t x : del) alive[x] = 0; t_del = now() - a;
            if ((faiss::idx_t)ins.front() != index.ntotal) { std::fprintf(stderr, "non-contiguous insert block\n"); return 2; }
            omp_set_num_threads(bth);
            a = now(); index.add((faiss::idx_t)ins.size(), X.data() + (size_t)ins.front() * d); t_ins = now() - a;
            for (int32_t x : ins) alive[x] = 1;
            cum += t_del + t_ins;
            if (e2e) { double b0 = now(); for (int32_t x : del) lmap[attr[x]][x] = 0; for (int32_t x : ins) lmap[attr[x]][x] = 1; t_pred = now() - b0; cum += t_pred; }
        }
        double t_rep = 0;
        if (repair && r > 0) {
            auto& g = index.acorn; const long nt0 = (long)index.ntotal; double a = now(); long nrep = 0;
            std::vector<double> m3v(4 * Lm, 0.0); double* m3p = m3v.data(); const int m3n = 4 * Lm; const bool m3on = mech && thr.abs > 0;
            auto elen = [&](long x, long y) { return std::sqrt(l2(X.data() + (size_t)x * d, X.data() + (size_t)y * d, d)); };
            omp_set_num_threads(bth);
            #pragma omp parallel for schedule(dynamic, 256) reduction(+:nrep) reduction(+:m3p[:m3n])
            for (long u = 0; u < nt0; ++u) {
                if (!alive[u]) continue;
                const int top = repair_mode >= 2 ? g.levels[u] : 1;     // levels[u] = number of levels of u
                for (int lev = (repair_mode == 3 ? 1 : 0); lev < top; ++lev) {
                size_t b, e; g.neighbor_range(u, lev, &b, &e);
                bool dead = false;
                for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (!alive[v]) { dead = true; break; } }
                if (!dead) continue;
                if (repair_mode == 4) {             // slot filling: keep live neighbors in order, refill dead slots
                    std::vector<int> live, repl; size_t ndead = 0;
                    for (size_t j = b; j < e; ++j) {
                        int v = g.neighbors[j]; if (v < 0) break;
                        if (alive[v]) { live.push_back(v); continue; }
                        ++ndead; size_t b2, e2; g.neighbor_range(v, lev, &b2, &e2); size_t lim = std::min(e2, b2 + (size_t)Mb);
                        for (size_t j2 = b2; j2 < lim; ++j2) { int w = g.neighbors[j2]; if (w < 0) break; if (alive[w] && w != u) repl.push_back(w); }
                    }
                    std::sort(repl.begin(), repl.end()); repl.erase(std::unique(repl.begin(), repl.end()), repl.end());
                    { std::vector<int> ls = live; std::sort(ls.begin(), ls.end());
                      repl.erase(std::remove_if(repl.begin(), repl.end(), [&](int w) { return std::binary_search(ls.begin(), ls.end(), w); }), repl.end()); }
                    std::vector<std::pair<float, int>> dr(repl.size()); const float* xu = X.data() + (size_t)u * d;
                    for (size_t i = 0; i < repl.size(); ++i) {
                        const float* xv = X.data() + (size_t)repl[i] * d; float s = 0;
                        for (int t = 0; t < d; ++t) { float df = xu[t] - xv[t]; s += df * df; }
                        dr[i] = {s, repl[i]};
                    }
                    size_t add = std::min(dr.size(), ndead); std::partial_sort(dr.begin(), dr.begin() + add, dr.end());
                    size_t j = b;
                    for (int v : live) g.neighbors[j++] = v;
                    for (size_t i = 0; i < add; ++i) g.neighbors[j++] = dr[i].second;
                    for (; j < e; ++j) g.neighbors[j] = -1;
                    if (m3on && lev == 0) { const int l = attr[u]; double nl = 0; for (int v : live) nl += elen(u, v) > thr.abs;
                        m3p[4 * l] += live.size(); m3p[4 * l + 1] += live.size(); m3p[4 * l + 2] += nl; m3p[4 * l + 3] += nl; }
                    ++nrep; continue;
                }
                std::vector<int> cand;
                for (size_t j = b; j < e; ++j) {
                    int v = g.neighbors[j]; if (v < 0) break;
                    if (alive[v]) { cand.push_back(v); continue; }
                    size_t b2, e2; g.neighbor_range(v, lev, &b2, &e2); size_t lim = std::min(e2, b2 + (size_t)Mb);
                    for (size_t j2 = b2; j2 < lim; ++j2) { int w = g.neighbors[j2]; if (w < 0) break; if (alive[w] && w != u) cand.push_back(w); }
                }
                std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                std::vector<std::pair<float, int>> dc(cand.size()); const float* xu = X.data() + (size_t)u * d;
                for (size_t i = 0; i < cand.size(); ++i) {
                    const float* xv = X.data() + (size_t)cand[i] * d; float s = 0;
                    for (int t = 0; t < d; ++t) { float df = xu[t] - xv[t]; s += df * df; }
                    dc[i] = {s, cand[i]};
                }
                size_t keep = std::min(dc.size(), e - b);
                std::partial_sort(dc.begin(), dc.begin() + keep, dc.end());
                if (m3on && lev == 0) {     // live neighbors before the repair, and how many the re-ranked list keeps
                    std::vector<int> lb; for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (alive[v]) lb.push_back(v); }
                    std::vector<int> kept; for (size_t j = 0; j < keep; ++j) kept.push_back(dc[j].second); std::sort(kept.begin(), kept.end());
                    const int l = attr[u]; double kb = 0, lbn = 0, kl = 0;
                    for (int v : lb) { bool lg = elen(u, v) > thr.abs; bool k_ = std::binary_search(kept.begin(), kept.end(), v); kb += k_; lbn += lg; kl += lg && k_; }
                    m3p[4 * l] += lb.size(); m3p[4 * l + 1] += kb; m3p[4 * l + 2] += lbn; m3p[4 * l + 3] += kl;
                }
                for (size_t j = 0; j < e - b; ++j) g.neighbors[b + j] = j < keep ? dc[j].second : -1;
                ++nrep;
                }
            }
            t_rep = now() - a; cum += t_rep;
            std::printf("{\"event\":\"repair\",\"round\":%d,\"nodes\":%ld,\"t_repair\":%.3f", r, nrep, t_rep);
            if (m3on) { std::printf(",\"m3\":["); for (int l = 0; l < Lm; ++l) std::printf("%s{\"label\":%d,\"live_before\":%.0f,\"kept\":%.0f,\"long_before\":%.0f,\"long_kept\":%.0f}", l ? "," : "", l, m3v[4*l], m3v[4*l+1], m3v[4*l+2], m3v[4*l+3]); std::printf("]"); }
            std::printf("}\n"); std::fflush(stdout);
        }
        const size_t nt = (size_t)index.ntotal;
        if (!measure_round(r)) {
            std::printf("{\"event\":\"round\",\"round\":%d,\"ntotal\":%zu,\"t_delete\":%.3f,\"t_insert\":%.3f,\"cum_maint_s\":%.3f}\n", r, nt, t_del, t_ins, cum);
            std::fflush(stdout); continue;
        }
        double rss_anon = 0, vm_rss = 0; rss_mb(rss_anon, vm_rss);     // before the per-query filter maps of the sweep are allocated
        std::vector<char> fmap(e2e ? 0 : (size_t)nq * nt);
        omp_set_num_threads(bth);
        if (!e2e) {
        #pragma omp parallel for
        for (int q = 0; q < nq; ++q)
            for (size_t x = 0; x < nt; ++x) fmap[(size_t)q * nt + x] = (char)(alive[x] && adm(q, x));
        }
        long live = 0; for (size_t x = 0; x < nt; ++x) live += alive[x];
        size_t gbytes = index.acorn.neighbors.size() * sizeof(index.acorn.neighbors[0]) + index.acorn.offsets.size() * sizeof(size_t)
                        + index.acorn.levels.size() * sizeof(int);   // as acorn_driver.cpp
        std::printf("{\"event\":\"round\",\"round\":%d,\"ntotal\":%zu,\"live\":%ld,\"t_delete\":%.3f,\"t_insert\":%.3f,\"cum_maint_s\":%.3f,\"graph_bytes\":%zu,\"graph_bytes_has_levels\":1,\"rss_anon_mb\":%.1f,\"vm_rss_mb\":%.1f",
                    r, nt, live, t_del, t_ins, cum, gbytes, rss_anon, vm_rss);
        omp_set_num_threads(sth);
        for (int ef : efs) {
            index.acorn.efSearch = ef;
            if (e2e) {
                std::vector<double> per(nq); std::string qs;
                for (int q = 0; q < std::min(nq, 32); ++q) index.search(1, Q.data() + (size_t)q * d, k, D.data() + (size_t)q * k, I.data() + (size_t)q * k, lmap[qbin[q]].data());
                for (int rep = 0; rep < reps; ++rep) {
                    double tot = 0;
                    for (int q = 0; q < nq; ++q) { double a = now(); index.search(1, Q.data() + (size_t)q * d, k, D.data() + (size_t)q * k, I.data() + (size_t)q * k, lmap[qbin[q]].data()); per[q] = now() - a; tot += per[q]; }
                    qs += (rep ? "," : "") + std::to_string(nq / tot);
                }
                std::vector<double> sp = per; std::sort(sp.begin(), sp.end()); auto pc = [&](double f) { return 1e3 * sp[std::min((size_t)(f * nq), (size_t)nq - 1)]; };
                std::vector<int32_t> o(I.size()); for (size_t i = 0; i < I.size(); ++i) o[i] = (int32_t)I[i];
                std::string p = out + "/acorn_r" + std::to_string(r) + "_ef" + std::to_string(ef) + ".i32";
                FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(o.data(), 4, o.size(), f); std::fclose(f);
                std::printf(",\"qps_ef%d\":[%s],\"lat_ef%d\":[%.4f,%.4f,%.4f]", ef, qs.c_str(), ef, pc(0.5), pc(0.95), pc(0.99));
                continue;
            }
            index.search(std::min(nq, 32), Q.data(), k, D.data(), I.data(), fmap.data());
            std::string qs;
            for (int rep = 0; rep < reps; ++rep) {
                double a = now(); index.search(nq, Q.data(), k, D.data(), I.data(), fmap.data()); double el = now() - a;
                qs += (rep ? "," : "") + std::to_string(nq / el);
            }
            std::vector<int32_t> o(I.size()); for (size_t i = 0; i < I.size(); ++i) o[i] = (int32_t)I[i];
            std::string p = out + "/acorn_r" + std::to_string(r) + "_ef" + std::to_string(ef) + ".i32";
            FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(o.data(), 4, o.size(), f); std::fclose(f);
            std::printf(",\"qps_ef%d\":[%s]", ef, qs.c_str());
        }
        std::printf(",\"t_repair\":%.3f,\"t_pred\":%.4f}\n", t_rep, t_pred); std::fflush(stdout);
        if (perc) {
            auto& g = index.acorn; const long ntl = (long)nt; const int cap = 2 * M;
            std::vector<std::vector<int>> eff(ntl);
            omp_set_num_threads(bth);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (long u = 0; u < ntl; ++u) {
                if (!alive[u]) continue;
                const int lab = attr[u]; std::vector<int>& out_u = eff[u];
                size_t b, e; g.neighbor_range(u, 0, &b, &e);
                for (size_t j = b; j < e && (int)out_u.size() < cap; ++j) {
                    int v = g.neighbors[j]; if (v < 0) break;
                    if (alive[v] && attr[v] == lab) out_u.push_back(v);
                    if ((int)(j - b) >= Mb || gamma == 1) {
                        size_t b2, e2; g.neighbor_range(v, 0, &b2, &e2);
                        for (size_t j2 = b2; j2 < e2 && (int)out_u.size() < cap; ++j2) {
                            int w = g.neighbors[j2]; if (w < 0) break;
                            if (w != u && alive[w] && attr[w] == lab) out_u.push_back(w);
                        }
                    }
                }
                std::sort(out_u.begin(), out_u.end()); out_u.erase(std::unique(out_u.begin(), out_u.end()), out_u.end());
            }
            std::vector<long> par(ntl); std::iota(par.begin(), par.end(), 0L);
            std::function<long(long)> find = [&](long x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
            for (long u = 0; u < ntl; ++u) for (int v : eff[u]) { long a = find(u), c = find(v); if (a != c) par[a] = c; }
            int L = 0; for (long u = 0; u < ntl; ++u) if (alive[u]) L = std::max(L, attr[u] + 1);
            std::vector<long> nlive(L, 0), nzero(L, 0), nupper(L, 0); std::vector<double> sdeg(L, 0);
            std::vector<std::vector<long>> roots(L);
            for (long u = 0; u < ntl; ++u) if (alive[u]) { int l = attr[u]; nlive[l]++; sdeg[l] += eff[u].size(); nzero[l] += eff[u].empty(); roots[l].push_back(find(u));
                                                            nupper[l] += g.levels[u] > 1; }   // live nodes of the label on level 1 or above (entries)
            std::printf("{\"event\":\"perc\",\"round\":%d,\"labels\":[", r);
            for (int l = 0; l < L; ++l) {
                std::sort(roots[l].begin(), roots[l].end()); long best = 0, ncomp = 0;
                for (size_t i = 0; i < roots[l].size();) { size_t j = i; while (j < roots[l].size() && roots[l][j] == roots[l][i]) ++j; best = std::max(best, (long)(j - i)); ++ncomp; i = j; }
                std::printf("%s{\"label\":%d,\"live\":%ld,\"deg_mean\":%.3f,\"deg0_frac\":%.5f,\"lcc_frac\":%.5f,\"components\":%ld,\"upper_live\":%ld}", l ? "," : "", l, nlive[l],
                            nlive[l] ? sdeg[l] / nlive[l] : 0.0, nlive[l] ? (double)nzero[l] / nlive[l] : 0.0, nlive[l] ? (double)best / nlive[l] : 0.0, ncomp, nupper[l]);
            }
            std::printf("]}\n"); std::fflush(stdout);
        }
        if (mech) {
            std::vector<int32_t> gt = read_i32(gtdir + "/gt_r" + std::to_string(r) + ".i32");
            mech_report(index, r, false, X.data(), d, alive, attr, gid_id, Q.data(), nq, qbin, gt, M, gamma, Mb, thr, bth, fmap);
            if (fresh && r == R) {          // a new index on the final live set, same parameters, ids mapped back
                std::vector<long> gid; for (size_t x = 0; x < nt; ++x) if (alive[x]) gid.push_back((long)x);
                const long nl = (long)gid.size(); std::vector<float> Xl((size_t)nl * d); std::vector<int32_t> al(nl); std::vector<uint8_t> alv(nl, 1);
                for (long i = 0; i < nl; ++i) { std::memcpy(Xl.data() + (size_t)i * d, X.data() + (size_t)gid[i] * d, d * sizeof(float)); al[i] = attr[gid[i]]; }
                std::vector<int> md(nl, 0); omp_set_num_threads(bth); double a = now();
                faiss::IndexACORNFlat fidx(d, M, gamma, md, Mb); fidx.add(nl, Xl.data()); double fb = now() - a;
                std::vector<char> fm((size_t)nq * nl);
                #pragma omp parallel for
                for (int q = 0; q < nq; ++q) for (long x = 0; x < nl; ++x) fm[(size_t)q * nl + x] = (char)(al[x] == qbin[q]);
                std::printf("{\"event\":\"fresh\",\"round\":%d,\"n\":%ld,\"build_s\":%.2f", r, nl, fb);
                omp_set_num_threads(sth);
                for (int ef : efs) {
                    fidx.acorn.efSearch = ef; std::string qs;
                    for (int rep = 0; rep < reps; ++rep) { double b0 = now(); fidx.search(nq, Q.data(), k, D.data(), I.data(), fm.data()); double el = now() - b0; qs += (rep ? "," : "") + std::to_string(nq / el); }
                    std::vector<int32_t> o(I.size()); for (size_t i = 0; i < I.size(); ++i) o[i] = I[i] >= 0 ? (int32_t)gid[I[i]] : -1;
                    std::string p = out + "/fresh_ef" + std::to_string(ef) + ".i32"; FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(o.data(), 4, o.size(), f); std::fclose(f);
                    std::printf(",\"qps_ef%d\":[%s]", ef, qs.c_str());
                }
                std::printf("}\n"); std::fflush(stdout);
                MechThr ft = thr; mech_report(fidx, r, true, Xl.data(), d, alv, al, gid, Q.data(), nq, qbin, gt, M, gamma, Mb, ft, bth, fm);
            }
        }
    }
    return 0;
}
