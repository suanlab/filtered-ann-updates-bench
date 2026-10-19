// acorn_stream.cpp — ACORN-gamma under an insert/delete stream with predicates (exp22).
// ACORN's own dynamic path: inserts via IndexACORN::add (incremental, on top of ntotal);
// deletes are applied by excluding the point from every query's filter map (alive AND predicate),
// i.e. tombstones that stay in the graph as routing nodes. Faiss ids equal Xall row ids because
// rows are added in increasing contiguous order (initial rows 0..N0-1, then each round's block).
//
// Usage: acorn_stream Xall.fbin Q.fbin attr.i32 qbin.i32 sched_dir N0 R M gamma M_beta
//                     build_threads search_threads efs_csv out_dir
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
        std::printf("}\n"); std::fflush(stdout);
    }
    return 0;
}
