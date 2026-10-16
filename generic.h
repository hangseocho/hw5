// generic.h — generic clamp/zscore, variadic sum, STL-based rolling statistics
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

// ---------- variadic template + fold expression ----------
// Type-safe sum: every argument must be arithmetic (checked at compile time),
// and the result type is the common type of all arguments.
template <typename... Ts>
constexpr auto sum(Ts... xs) {
    static_assert(sizeof...(Ts) > 0, "sum() needs at least one argument");
    static_assert((std::is_arithmetic_v<Ts> && ...), "sum() accepts arithmetic types only");
    using R = std::common_type_t<Ts...>;
    return (R{0} + ... + static_cast<R>(xs));   // binary left fold
}

// ---------- generic clamp, constrained with SFINAE ----------
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
constexpr T clamp_val(T x, T lo, T hi) {
    return x < lo ? lo : (hi < x ? hi : x);
}

// ---------- generic zscore, branching with if constexpr ----------
// Integers are promoted to double; floating types keep their own precision.
template <typename T>
auto zscore(T x, T mean, T sd) {
    if constexpr (std::is_integral_v<T>) {
        const double s = static_cast<double>(sd);
        return s == 0.0 ? 0.0 : (static_cast<double>(x) - mean) / s;
    } else if constexpr (std::is_floating_point_v<T>) {
        return sd == T{0} ? T{0} : (x - mean) / sd;
    } else {
        static_assert(sizeof(T) == 0, "zscore requires an arithmetic type");
    }
}

// ---------- STL pipeline: prices -> returns -> rolling stats ----------
template <typename T>
std::vector<T> log_returns(const std::vector<T>& prices) {
    static_assert(std::is_floating_point_v<T>, "log_returns needs a floating type");
    if (prices.size() < 2) return {};
    std::vector<T> r(prices.size() - 1);
    // r[i] = log(p[i+1] / p[i])  — binary std::transform over two shifted ranges
    std::transform(prices.begin() + 1, prices.end(), prices.begin(), r.begin(),
                   [](T next, T prev) { return std::log(next / prev); });
    return r;
}

template <typename T>
struct RollingStats {
    std::vector<T> mean, stddev, median;
};

// Naive rolling window: recompute each window with STL algorithms. O(n * w log w).
template <typename T>
RollingStats<T> rolling_stats_stl(const std::vector<T>& x, std::size_t w) {
    RollingStats<T> out;
    if (w < 2 || x.size() < w) return out;
    const std::size_t n = x.size() - w + 1;
    out.mean.reserve(n); out.stddev.reserve(n); out.median.reserve(n);
    std::vector<T> scratch(w);

    for (std::size_t i = 0; i < n; ++i) {
        auto first = x.begin() + i, last = first + w;
        const T m = std::accumulate(first, last, T{0}) / static_cast<T>(w);
        // sum of squared deviations via transform_reduce-style accumulate
        const T ss = std::accumulate(first, last, T{0},
                                     [m](T acc, T v) { return acc + (v - m) * (v - m); });
        std::copy(first, last, scratch.begin());
        std::sort(scratch.begin(), scratch.end());
        const T med = (w % 2) ? scratch[w / 2] : (scratch[w / 2 - 1] + scratch[w / 2]) / T{2};

        out.mean.push_back(m);
        out.stddev.push_back(std::sqrt(ss / static_cast<T>(w - 1)));
        out.median.push_back(med);
    }
    return out;
}
