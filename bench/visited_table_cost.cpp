// Cost of ACORN's per-call visited table (faiss::VisitedTable(ntotal): std::vector<uint8_t>(ntotal, 0)), which its
// one-query-per-call mode pays inside the timer; median over 200 constructions with a few random touches.
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::strtoull(argv[1], 0, 10) : 10000000; std::vector<double> t;
    for (int r = 0; r < 200; ++r) {
        auto a = std::chrono::steady_clock::now();
        { std::vector<uint8_t> v(n, 0); for (int i = 0; i < 1000; ++i) v[(size_t)rand() % n] = 1; asm volatile("" :: "r"(v.data()) : "memory"); }
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end()); std::printf("{\"n\": %zu, \"median_s\": %.6g, \"p90_s\": %.6g}\n", n, t[100], t[180]);
}
