// cc_ivf.cpp -- concurrent maintenance and search for IVF (follow-up 2, I18c). Faiss (ACORN's fork, 1.7.x) in C++,
// because the Python binding of the study's Faiss does not release the GIL during search.
//
// Usage: cc_ivf <dir> <ivf|ivf_part> <N0> <R> <readers> <writer_threads> <batch>
//   <dir> holds X.fbin, Q.fbin, attr.i32, qlab.i32, del<r>.i32, ins<r>.i32, gt0.i32 (round-0 ground truth, nq x 10).
//   ivf      one IVF-Flat (nlist = 4 sqrt(N0)), per-label packed bitmaps as IDSelectorBitmap, one reader-writer lock;
//   ivf_part one IVF-Flat per label (nlist = 4 sqrt(label size)), one reader-writer lock per label, no predicate.
// Round 0: the nprobe is chosen on the quiet index as the smallest of 1,2,4,...,1024 with recall >= 0.95 and kept.
// Each round r >= 1: one writer applies the round's deletions and insertions in alternating batches of <batch> points
// (each batch under the write lock(s) it touches), while <readers> threads issue the 500 queries in a fixed shuffled
// cycle until the writer is done; then every reader runs the queries once more with no writer (quiet phase).
// Output (<dir>/out): per round, cc_r<r>.bin with one record per query execution
//   {int32 qid, int32 phase (0 concurrent, 1 quiet), double t_start, double t_end, int32 ids[10]},
// and del_time_r<r>.bin with (int32 id, double t) for every deleted id (time its batch's lock was released).
// Prints one JSON line per round with nprobe, writer seconds, and counts.
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

static std::vector<float> read_fbin(const std::string& p, int& n, int& d) {
    FILE* f = std::fopen(p.c_str(), "rb"); if (!f) { std::perror(p.c_str()); std::exit(1); }
    int32_t h[2]; if (std::fread(h, 4, 2, f) != 2) std::exit(1); n = h[0]; d = h[1];
    std::vector<float> v((size_t)n * d); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1); std::fclose(f); return v;
}
static std::vector<int32_t> read_i32(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb"); if (!f) { std::perror(p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(1); std::fclose(f); return v;
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Rec { int32_t q, phase; double t0, t1; int32_t ids[10]; };
// fair reader-writer lock (turnstile): a waiting writer blocks new readers, so writers do not starve
struct FairRW {
    std::mutex turn; std::shared_mutex rw;
    void lock_shared() { { std::lock_guard<std::mutex> t(turn); } rw.lock_shared(); }
    void unlock_shared() { rw.unlock_shared(); }
    void lock() { turn.lock(); rw.lock(); turn.unlock(); }
    void unlock() { rw.unlock(); }
};

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: see header\n"); return 1; }
    std::string dir = argv[1], mode = argv[2]; const int N0 = std::atoi(argv[3]), R = std::atoi(argv[4]);
    const int NR = std::atoi(argv[5]), WT = std::atoi(argv[6]), BATCH = std::atoi(argv[7]); const bool part = mode == "ivf_part";
    int N, d, nq, dq; std::vector<float> X = read_fbin(dir + "/X.fbin", N, d), Q = read_fbin(dir + "/Q.fbin", nq, dq);
    std::vector<int32_t> attr = read_i32(dir + "/attr.i32"), qlab = read_i32(dir + "/qlab.i32"), gt0 = read_i32(dir + "/gt0.i32");
    int L = 0; for (int32_t a : attr) L = std::max(L, a + 1);
    const int k = 10; omp_set_num_threads(WT);
    // ---- build
    std::vector<std::unique_ptr<faiss::IndexFlatL2>> quants; std::vector<std::unique_ptr<faiss::IndexIVFFlat>> ivfs;
    std::vector<std::vector<uint8_t>> bits(part ? 0 : L, std::vector<uint8_t>((N + 7) / 8, 0));
    std::vector<std::unique_ptr<faiss::IDSelectorBitmap>> sels;
    std::vector<FairRW> locks(part ? L : 1);
    std::mt19937_64 rng(0);
    auto build_one = [&](const std::vector<faiss::idx_t>& ids) {
        int nlist = std::max(1, (int)(4 * std::sqrt((double)ids.size())));
        quants.emplace_back(new faiss::IndexFlatL2(d)); ivfs.emplace_back(new faiss::IndexIVFFlat(quants.back().get(), d, nlist));
        std::vector<faiss::idx_t> s = ids; std::shuffle(s.begin(), s.end(), rng); s.resize(std::min(s.size(), (size_t)50 * nlist));
        std::vector<float> tr(s.size() * d); for (size_t i = 0; i < s.size(); ++i) std::memcpy(&tr[i * d], &X[(size_t)s[i] * d], d * 4);
        ivfs.back()->train(s.size(), tr.data());
        std::vector<float> xb(ids.size() * d); for (size_t i = 0; i < ids.size(); ++i) std::memcpy(&xb[i * d], &X[(size_t)ids[i] * d], d * 4);
        ivfs.back()->add_with_ids(ids.size(), xb.data(), ids.data());
    };
    double tb = now();
    if (part) { for (int l = 0; l < L; ++l) { std::vector<faiss::idx_t> ids; for (int i = 0; i < N0; ++i) if (attr[i] == l) ids.push_back(i); build_one(ids); } }
    else { std::vector<faiss::idx_t> ids(N0); for (int i = 0; i < N0; ++i) ids[i] = i; build_one(ids);
           for (int i = 0; i < N0; ++i) bits[attr[i]][i >> 3] |= (uint8_t)(1 << (i & 7));
           for (int l = 0; l < L; ++l) sels.emplace_back(new faiss::IDSelectorBitmap(N, bits[l].data())); }
    std::printf("{\"event\":\"build\",\"build_s\":%.2f}\n", now() - tb); std::fflush(stdout);
    auto search = [&](int q, int nprobe, faiss::idx_t* I, float* D) {
        const int l = qlab[q];
        if (part) { std::shared_lock<FairRW> g(locks[l]); faiss::SearchParametersIVF p; p.nprobe = std::min(nprobe, (int)ivfs[l]->nlist); ivfs[l]->search(1, &Q[(size_t)q * d], k, D, I, &p); }
        else { std::shared_lock<FairRW> g(locks[0]); faiss::SearchParametersIVF p; p.nprobe = nprobe; p.sel = sels[l].get(); ivfs[0]->search(1, &Q[(size_t)q * d], k, D, I, &p); }
    };
    // ---- round 0: choose nprobe on the quiet index
    omp_set_num_threads(1); int nprobe = 1024;
    for (int np_ = 1; np_ <= 1024; np_ *= 2) {
        double hit = 0, tot = 0; std::vector<faiss::idx_t> I(k); std::vector<float> D(k);
        for (int q = 0; q < nq; ++q) { search(q, np_, I.data(), D.data()); int ng = 0, h = 0;
            for (int j = 0; j < k; ++j) { int g_ = gt0[(size_t)q * k + j]; if (g_ < 0) continue; ++ng; for (int t = 0; t < k; ++t) if (I[t] == g_) { ++h; break; } }
            hit += ng ? (double)h / ng : 1; tot += 1; }
        if (hit / tot >= 0.95) { nprobe = np_; break; }
    }
    std::printf("{\"event\":\"calibrate\",\"nprobe\":%d}\n", nprobe); std::fflush(stdout);
    std::vector<int> cycle(nq); for (int i = 0; i < nq; ++i) cycle[i] = i; std::shuffle(cycle.begin(), cycle.end(), std::mt19937_64(7));
    std::system(("mkdir -p " + dir + "/out").c_str());
    for (int r = 1; r <= R; ++r) {
        std::vector<int32_t> del = read_i32(dir + "/del" + std::to_string(r - 1) + ".i32"), ins = read_i32(dir + "/ins" + std::to_string(r - 1) + ".i32");
        std::vector<std::pair<int32_t, double>> dtime; dtime.reserve(del.size());
        std::atomic<bool> done(false); std::atomic<long> next(0); std::vector<std::vector<Rec>> recs(NR);
        auto reader = [&](int t, int phase, long limit) {
            omp_set_num_threads(1); std::vector<faiss::idx_t> I(k); std::vector<float> D(k);
            for (;;) {
                if (phase == 0 && done.load()) break;
                long c = next.fetch_add(1); if (phase == 1 && c >= limit) break;
                int q = cycle[c % nq]; Rec rc; rc.q = q; rc.phase = phase; rc.t0 = now(); search(q, nprobe, I.data(), D.data()); rc.t1 = now();
                for (int j = 0; j < k; ++j) rc.ids[j] = (int32_t)I[j]; recs[t].push_back(rc);
            }
        };
        double tw0 = now(), twall = 0;
        std::thread writer([&] {
            omp_set_num_threads(WT); size_t di = 0, ii = 0;
            while (di < del.size() || ii < ins.size()) {
                if (di < del.size()) {           // a batch of deletions
                    size_t e = std::min(del.size(), di + BATCH); std::vector<faiss::idx_t> b(del.begin() + di, del.begin() + e);
                    if (part) {
                        for (int l = 0; l < L; ++l) { std::vector<faiss::idx_t> bl; for (auto x : b) if (attr[x] == l) bl.push_back(x); if (bl.empty()) continue;
                            std::unique_lock<FairRW> g(locks[l]); faiss::IDSelectorBatch s(bl.size(), bl.data()); ivfs[l]->remove_ids(s); }
                    } else {
                        std::unique_lock<FairRW> g(locks[0]); faiss::IDSelectorBatch s(b.size(), b.data()); ivfs[0]->remove_ids(s);
                        for (auto x : b) bits[attr[x]][x >> 3] &= (uint8_t)~(1 << (x & 7));
                    }
                    double t = now(); for (auto x : b) dtime.push_back({(int32_t)x, t}); di = e;
                }
                if (ii < ins.size()) {           // a batch of insertions
                    size_t e = std::min(ins.size(), ii + BATCH); std::vector<faiss::idx_t> b(ins.begin() + ii, ins.begin() + e);
                    if (part) {
                        for (int l = 0; l < L; ++l) { std::vector<faiss::idx_t> bl; for (auto x : b) if (attr[x] == l) bl.push_back(x); if (bl.empty()) continue;
                            std::vector<float> xb(bl.size() * d); for (size_t i = 0; i < bl.size(); ++i) std::memcpy(&xb[i * d], &X[(size_t)bl[i] * d], d * 4);
                            std::unique_lock<FairRW> g(locks[l]); ivfs[l]->add_with_ids(bl.size(), xb.data(), bl.data()); }
                    } else {
                        std::vector<float> xb(b.size() * d); for (size_t i = 0; i < b.size(); ++i) std::memcpy(&xb[i * d], &X[(size_t)b[i] * d], d * 4);
                        std::unique_lock<FairRW> g(locks[0]); ivfs[0]->add_with_ids(b.size(), xb.data(), b.data());
                        for (auto x : b) bits[attr[x]][x >> 3] |= (uint8_t)(1 << (x & 7));
                    }
                    ii = e;
                }
            }
            twall = now() - tw0; done.store(true);
        });
        std::vector<std::thread> rd; for (int t = 0; t < NR; ++t) rd.emplace_back(reader, t, 0, 0);
        writer.join(); for (auto& t : rd) t.join(); long nconc = next.load();
        next.store(0); rd.clear(); for (int t = 0; t < NR; ++t) rd.emplace_back(reader, t, 1, (long)nq); for (auto& t : rd) t.join();
        std::string p = dir + "/out/cc_r" + std::to_string(r) + ".bin"; FILE* f = std::fopen(p.c_str(), "wb"); long nrec = 0;
        for (auto& v : recs) { std::fwrite(v.data(), sizeof(Rec), v.size(), f); nrec += v.size(); } std::fclose(f);
        p = dir + "/out/del_time_r" + std::to_string(r) + ".bin"; f = std::fopen(p.c_str(), "wb");
        for (auto& pr : dtime) { std::fwrite(&pr.first, 4, 1, f); std::fwrite(&pr.second, 8, 1, f); } std::fclose(f);
        std::printf("{\"event\":\"round\",\"round\":%d,\"nprobe\":%d,\"writer_s\":%.3f,\"concurrent_queries\":%ld,\"records\":%ld,\"t_writer_start\":%.6f}\n", r, nprobe, twall, nconc, nrec, tw0);
        std::fflush(stdout);
    }
    return 0;
}
