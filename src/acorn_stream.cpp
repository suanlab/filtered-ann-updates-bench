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
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

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
    const bool perc = std::getenv("PERC") && std::atoi(std::getenv("PERC")) > 0;
    const int repair_mode = std::getenv("REPAIR") ? std::atoi(std::getenv("REPAIR")) : 0; const bool repair = repair_mode > 0;
    std::vector<int> metadata(N, 0);                 // sized to full capacity: never indexed out of range
    std::vector<uint8_t> alive(N, 0);
    omp_set_num_threads(bth);
    double t0 = now();
    faiss::IndexACORNFlat index(d, M, gamma, metadata, Mb);
    index.add(N0, X.data());
    for (int i = 0; i < N0; ++i) alive[i] = 1;
    std::printf("{\"event\":\"build\",\"n\":%d,\"build_s\":%.2f}\n", N0, now() - t0); std::fflush(stdout);
    const int k = 10;
    std::vector<float> D((size_t)nq * k); std::vector<faiss::idx_t> I((size_t)nq * k);
    double cum = 0;
    for (int r = 0; r <= R; ++r) {
        double t_del = 0, t_ins = 0;
        if (r > 0) {
            std::vector<int32_t> del = read_i32(sched + "/del" + std::to_string(r - 1) + ".i32");
            std::vector<int32_t> ins = read_i32(sched + "/ins" + std::to_string(r - 1) + ".i32");
            double a = now(); for (int32_t x : del) alive[x] = 0; t_del = now() - a;
            if ((faiss::idx_t)ins.front() != index.ntotal) { std::fprintf(stderr, "non-contiguous insert block\n"); return 2; }
            omp_set_num_threads(bth);
            a = now(); index.add((faiss::idx_t)ins.size(), X.data() + (size_t)ins.front() * d); t_ins = now() - a;
            for (int32_t x : ins) alive[x] = 1;
            cum += t_del + t_ins;
        }
        double t_rep = 0;
        if (repair && r > 0) {
            auto& g = index.acorn; const long nt0 = (long)index.ntotal; double a = now(); long nrep = 0;
            omp_set_num_threads(bth);
            #pragma omp parallel for schedule(dynamic, 256) reduction(+:nrep)
            for (long u = 0; u < nt0; ++u) {
                if (!alive[u]) continue;
                const int top = repair_mode >= 2 ? g.levels[u] : 1;     // levels[u] = number of levels of u
                for (int lev = 0; lev < top; ++lev) {
                size_t b, e; g.neighbor_range(u, lev, &b, &e);
                bool dead = false;
                for (size_t j = b; j < e; ++j) { int v = g.neighbors[j]; if (v < 0) break; if (!alive[v]) { dead = true; break; } }
                if (!dead) continue;
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
                for (size_t j = 0; j < e - b; ++j) g.neighbors[b + j] = j < keep ? dc[j].second : -1;
                ++nrep;
                }
            }
            t_rep = now() - a; cum += t_rep;
            std::printf("{\"event\":\"repair\",\"round\":%d,\"nodes\":%ld,\"t_repair\":%.3f}\n", r, nrep, t_rep); std::fflush(stdout);
        }
        const size_t nt = (size_t)index.ntotal;
        if (!measure_round(r)) {
            std::printf("{\"event\":\"round\",\"round\":%d,\"ntotal\":%zu,\"t_delete\":%.3f,\"t_insert\":%.3f,\"cum_maint_s\":%.3f}\n", r, nt, t_del, t_ins, cum);
            std::fflush(stdout); continue;
        }
        std::vector<char> fmap((size_t)nq * nt);
        omp_set_num_threads(bth);
        #pragma omp parallel for
        for (int q = 0; q < nq; ++q)
            for (size_t x = 0; x < nt; ++x) fmap[(size_t)q * nt + x] = (char)(alive[x] && attr[x] == qbin[q]);
        long live = 0; for (size_t x = 0; x < nt; ++x) live += alive[x];
        size_t gbytes = index.acorn.neighbors.size() * sizeof(index.acorn.neighbors[0]) + index.acorn.offsets.size() * sizeof(size_t)
                        + index.acorn.levels.size() * sizeof(int);   // as acorn_driver.cpp
        std::printf("{\"event\":\"round\",\"round\":%d,\"ntotal\":%zu,\"live\":%ld,\"t_delete\":%.3f,\"t_insert\":%.3f,\"cum_maint_s\":%.3f,\"graph_bytes\":%zu,\"graph_bytes_has_levels\":1",
                    r, nt, live, t_del, t_ins, cum, gbytes);
        omp_set_num_threads(sth);
        for (int ef : efs) {
            index.acorn.efSearch = ef;
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
        std::printf(",\"t_repair\":%.3f}\n", t_rep); std::fflush(stdout);
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
    }
    return 0;
}
