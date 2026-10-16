// main.cpp — tests + benchmark for Week 5 (Templates & Generic Programming)
// Build: g++ -std=c++17 -O2 -Wall -Wextra -o hw5 main.cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "generic.h"
#include "ringstat.h"

static bool close(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

// Synthetic price path: geometric Brownian motion, fixed seed for reproducibility.
static std::vector<double> make_prices(std::size_t n, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    const double mu = 0.0002, sigma = 0.01;
    std::vector<double> p(n);
    p[0] = 100.0;
    for (std::size_t i = 1; i < n; ++i)
        p[i] = p[i - 1] * std::exp(mu - 0.5 * sigma * sigma + sigma * z(rng));
    return p;
}

static void run_tests() {
    // sum: fold expression, mixed types promote to common type
    static_assert(sum(1, 2, 3) == 6);
    static_assert(std::is_same_v<decltype(sum(1, 2.5)), double>);
    assert(close(sum(1, 2.5f, 3.0), 6.5));
    // sum("a", 1);  // <- does not compile: static_assert "arithmetic types only"

    // clamp_val (SFINAE)
    static_assert(clamp_val(5, 0, 3) == 3);
    static_assert(clamp_val(-1.0, 0.0, 1.0) == 0.0);
    // clamp_val(std::string("a"), ...);  // <- no matching function (SFINAE removes it)

    // zscore (if constexpr): int promotes to double, float stays float
    static_assert(std::is_same_v<decltype(zscore(3, 1, 2)), double>);
    static_assert(std::is_same_v<decltype(zscore(3.f, 1.f, 2.f)), float>);
    assert(close(zscore(3, 1, 2), 1.0));
    assert(zscore(5.0, 5.0, 0.0) == 0.0);  // zero-sd guard

    // RingStat basics
    RingStat<int> ri(3);
    for (int v : {1, 2, 3, 4}) ri.push(v);  // window = {2,3,4}
    assert(ri.full() && close(ri.mean(), 3.0) && close(ri.variance(), 1.0));
    // RingStat<std::string> rs(3);  // <- does not compile (enable_if on class)

    // RingStat must agree with the STL recomputation on real data
    auto prices = make_prices(2000);
    auto r = log_returns(prices);
    const std::size_t w = 20;
    auto ref = rolling_stats_stl(r, w);
    RingStat<double> rs(w);
    std::size_t k = 0;
    double max_err = 0;
    for (std::size_t i = 0; i < r.size(); ++i) {
        rs.push(r[i]);
        if (rs.full()) {
            max_err = std::max(max_err, std::fabs(rs.mean() - ref.mean[k]));
            max_err = std::max(max_err, std::fabs(rs.stddev() - ref.stddev[k]));
            ++k;
        }
    }
    assert(k == ref.mean.size());
    assert(max_err < 1e-12);
    std::printf("All tests passed. (RingStat vs STL max abs error = %.2e over %zu windows)\n",
                max_err, k);
}

template <typename F>
static double time_ms(F&& f, int reps = 5) {
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;  // best-of-N reduces noise from the OS
}

static void run_benchmark() {
    const std::size_t n = 1'000'000;
    auto prices = make_prices(n + 1);
    auto r = log_returns(prices);

    std::printf("\nBenchmark: rolling mean+stddev over %zu daily log-returns (best of 5)\n", r.size());
    std::printf("%8s | %14s | %16s | %16s | %8s\n", "window", "RingStat (ms)",
                "STL mean+sd (ms)", "STL +median (ms)", "speedup");

    for (std::size_t w : {5, 20, 60, 252}) {
        volatile double sink = 0;  // prevent dead-code elimination

        double t_ring = time_ms([&] {
            RingStat<double> rs(w);
            double acc = 0;
            for (double x : r) { rs.push(x); if (rs.full()) acc += rs.mean() + rs.stddev(); }
            sink = acc;
        });

        double t_stl_ms = time_ms([&] {  // accumulate-only (no sort) for a fair comparison
            double acc = 0;
            for (std::size_t i = 0; i + w <= r.size(); ++i) {
                auto f = r.begin() + i, l = f + w;
                double m = std::accumulate(f, l, 0.0) / w;
                double ss = std::accumulate(f, l, 0.0, [m](double a, double v) { return a + (v - m) * (v - m); });
                acc += m + std::sqrt(ss / (w - 1));
            }
            sink = acc;
        });

        double t_stl_full = time_ms([&] {
            auto s = rolling_stats_stl(r, w);
            sink = s.median.back();
        }, 1);

        (void)sink;
        std::printf("%8zu | %14.2f | %16.2f | %16.2f | %7.1fx\n", w, t_ring, t_stl_ms, t_stl_full,
                    t_stl_ms / t_ring);
    }

    // Annualized summary of the full sample
    RingStat<double> all(r.size());
    for (double x : r) all.push(x);
    std::printf("\nFull-sample: mean daily log-return = %.6f, daily sd = %.6f, annualized vol = %.2f%%\n",
                all.mean(), all.stddev(), all.stddev() * std::sqrt(252.0) * 100);
}

int main() {
    run_tests();
    run_benchmark();
}
