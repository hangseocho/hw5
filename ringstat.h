// ringstat.h — generic fixed-window rolling statistics (O(1) per update)
#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

// SFINAE: RingStat<T> only exists for arithmetic T (int, float, double, ...).
// RingStat<std::string> fails to compile with a clear "no type named 'type'" error.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class RingStat {
public:
    // Accumulate in double for integer T (avoids overflow/truncation),
    // in long double for long double, otherwise in T itself.
    using acc_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

    explicit RingStat(std::size_t window) : buf_(window), cap_(window) {
        if (window == 0) throw std::invalid_argument("RingStat: window must be > 0");
    }

    void push(T x) {
        if (size_ == cap_) {               // evict oldest
            const acc_t old = static_cast<acc_t>(buf_[head_]);
            sum_ -= old;
            sumsq_ -= old * old;
        } else {
            ++size_;
        }
        buf_[head_] = x;
        const acc_t v = static_cast<acc_t>(x);
        sum_ += v;
        sumsq_ += v * v;
        head_ = (head_ + 1) % cap_;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return cap_; }
    bool full() const noexcept { return size_ == cap_; }

    acc_t mean() const { return size_ ? sum_ / static_cast<acc_t>(size_) : acc_t{0}; }

    // Sample variance (n-1). Clamped at 0 to absorb floating-point cancellation.
    acc_t variance() const {
        if (size_ < 2) return acc_t{0};
        const acc_t n = static_cast<acc_t>(size_);
        const acc_t var = (sumsq_ - sum_ * sum_ / n) / (n - 1);
        return var > 0 ? var : acc_t{0};
    }
    acc_t stddev() const { return std::sqrt(variance()); }

private:
    std::vector<T> buf_;
    std::size_t cap_, head_ = 0, size_ = 0;
    acc_t sum_{0}, sumsq_{0};
};
