// acorn_driver.cpp — runs the real ACORN-gamma index (stanford-futuredata/ACORN, SIGMOD'24)
// on our predicate workloads. The ACORN graph is predicate-agnostic, so it is built once and
// searched under several attribute/predicate sets.
//
// Usage: acorn_driver base.fbin query.fbin M gamma M_beta build_threads:search_threads efs_csv out_dir  job1 [job2 ...]
//   job = attr_file:qbin_file:tag   (attr_file: int32[N] bin id per base point,
//                                     qbin_file: int32[nq] required bin per query)
// Writes out_dir/acorn_<tag>_ef<ef>.i32 (nq x 10 ids) and prints build/search timings as JSON lines.
#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h>
#ifdef ACORN_COUNT_CHECKS     // counting build (third_party/ACORN/build_count): level-0 hops and predicate checks
namespace faiss { void acorn_count_reset(); size_t acorn_count_get_hops(); size_t acorn_count_get_checks(); }
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <omp.h>

static std::vector<float> read_fbin(const char* path, int& n, int& d) {
    FILE* f = std::fopen(path, "rb"); if (!f) { std::perror(path); std::exit(1); }
    int32_t hdr[2]; if (std::fread(hdr, 4, 2, f) != 2) { std::exit(1); }
    n = hdr[0]; d = hdr[1];
    std::vector<float> v((size_t)n * d);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    std::fclose(f); return v;
}
static std::vector<int32_t> read_i32(const char* path, size_t expect) {
    FILE* f = std::fopen(path, "rb"); if (!f) { std::perror(path); std::exit(1); }
    std::vector<int32_t> v(expect);
    if (std::fread(v.data(), 4, expect, f) != expect) { std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
    std::fclose(f); return v;
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 10) { std::fprintf(stderr, "usage: see header\n"); return 1; }
    int N, d, nq, dq;
    std::vector<float> xb = read_fbin(argv[1], N, d);
    std::vector<float> xq = read_fbin(argv[2], nq, dq);
    int M = std::atoi(argv[3]), gamma = std::atoi(argv[4]), M_beta = std::atoi(argv[5]);
    int threads = std::atoi(argv[6]); const char* colon = std::strchr(argv[6], ':');
    int sthreads = colon ? std::atoi(colon + 1) : threads;
    std::vector<int> efs; { std::stringstream ss(argv[7]); std::string t; while (std::getline(ss, t, ',')) efs.push_back(std::atoi(t.c_str())); }
    std::string outdir = argv[8];
    omp_set_num_threads(threads);
    std::vector<int> metadata(N, 0);   // ACORN-gamma construction is predicate-agnostic
    double t0 = now();
    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    index.add(N, xb.data());
    double tb = now() - t0;
    size_t graph_bytes = index.acorn.neighbors.size() * sizeof(index.acorn.neighbors[0]) + index.acorn.offsets.size() * sizeof(size_t)
                       + index.acorn.levels.size() * sizeof(int);
    std::printf("{\"event\":\"build\",\"N\":%d,\"d\":%d,\"M\":%d,\"gamma\":%d,\"M_beta\":%d,\"build_s\":%.2f,\"graph_bytes\":%zu}\n", N, d, M, gamma, M_beta, tb, graph_bytes);
    std::fflush(stdout);
    const int k = 10;
    std::vector<char> fmap((size_t)nq * N);
    std::printf("{\"event\":\"threads\",\"build\":%d,\"search\":%d}\n", threads, sthreads);
    std::vector<float> D((size_t)nq * k); std::vector<faiss::idx_t> I((size_t)nq * k);
    for (int j = 9; j < argc; ++j) {
        std::string job = argv[j];
        bool csr = job.rfind("csr:", 0) == 0;          // csr:<qoff.i64>:<qids.i32>:<tag>  (arbitrary admissible sets)
        if (csr) job = job.substr(4);                   // else <attr.i32>:<qlab.i32>:<tag> (label equality)
        size_t a = job.find(':'), b = job.find(':', a + 1);
        std::string f1 = job.substr(0, a), f2 = job.substr(a + 1, b - a - 1), tag = job.substr(b + 1);
        omp_set_num_threads(threads);
        if (csr) {
            FILE* fo = std::fopen(f1.c_str(), "rb"); std::vector<int64_t> qoff(nq + 1);
            if (!fo || std::fread(qoff.data(), 8, nq + 1, fo) != (size_t)(nq + 1)) { std::fprintf(stderr, "bad qoff\n"); return 1; }
            std::fclose(fo);
            std::vector<int32_t> qids = read_i32(f2.c_str(), (size_t)qoff[nq]);
            std::fill(fmap.begin(), fmap.end(), 0);
            #pragma omp parallel for
            for (int q = 0; q < nq; ++q)
                for (int64_t t = qoff[q]; t < qoff[q + 1]; ++t) fmap[(size_t)q * N + qids[t]] = 1;
        } else {
            std::vector<int32_t> attr = read_i32(f1.c_str(), N), qbin = read_i32(f2.c_str(), nq);
            #pragma omp parallel for
            for (int q = 0; q < nq; ++q)
                for (int x = 0; x < N; ++x) fmap[(size_t)q * N + x] = (char)(attr[x] == qbin[q]);
        }
        const char* re = std::getenv("ACORN_REPS"); int reps = re ? std::max(1, std::atoi(re)) : 1;
        omp_set_num_threads(sthreads);
        for (int ef : efs) {
            index.acorn.efSearch = ef;
            index.search(std::min(nq, 64), xq.data(), k, D.data(), I.data(), fmap.data());   // warmup
            std::string qs; std::vector<double> tq(nq, 0.0);
            faiss::acorn_stats.reset();      // level-0 searches (n1) and their distance computations (n3), all reps
#ifdef ACORN_COUNT_CHECKS
            faiss::acorn_count_reset();
#endif
            for (int r = 0; r < reps; ++r) {
                double el;
                if (csr) {          // one query per call so that per-query times are available
                    el = 0;
                    for (int q = 0; q < nq; ++q) {
                        double ts = now();
                        index.search(1, xq.data() + (size_t)q * d, k, D.data() + (size_t)q * k, I.data() + (size_t)q * k, fmap.data() + (size_t)q * N);
                        tq[q] = now() - ts; el += tq[q];
                    }
                } else {
                    double ts = now();
                    index.search(nq, xq.data(), k, D.data(), I.data(), fmap.data());
                    el = now() - ts;
                }
                qs += (r ? "," : "") + std::to_string(nq / el);
            }
            if (csr) {
                std::string tp = outdir + "/acorn_" + tag + "_ef" + std::to_string(ef) + ".t";
                FILE* ft = std::fopen(tp.c_str(), "wb"); std::fwrite(tq.data(), 8, nq, ft); std::fclose(ft);
            }
            std::string path = outdir + "/acorn_" + tag + "_ef" + std::to_string(ef) + ".i32";
            FILE* f = std::fopen(path.c_str(), "wb");
            std::vector<int32_t> o(I.size()); for (size_t i = 0; i < I.size(); ++i) o[i] = (int32_t)I[i];
            std::fwrite(o.data(), 4, o.size(), f); std::fclose(f);
            double ndq = faiss::acorn_stats.n1 ? (double)faiss::acorn_stats.n3 / faiss::acorn_stats.n1 : -1.0;
#ifdef ACORN_COUNT_CHECKS
            double nsr = (double)nq * reps;
            std::printf("{\"event\":\"counts\",\"tag\":\"%s\",\"ef\":%d,\"hops_per_query\":%.2f,\"checks_per_query\":%.2f}\n", tag.c_str(), ef,
                        faiss::acorn_count_get_hops() / nsr, faiss::acorn_count_get_checks() / nsr);
#endif
            std::printf("{\"event\":\"search\",\"tag\":\"%s\",\"ef\":%d,\"qps_reps\":[%s],\"ndis_per_query\":%.2f}\n", tag.c_str(), ef, qs.c_str(), ndq);
            std::fflush(stdout);
        }
    }
    return 0;
}
