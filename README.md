# Week 5 — Templates & Generic Programming

## Files
| File | Contents |
|---|---|
| `ringstat.h` | `RingStat<T>` — fixed-window ring buffer with O(1) rolling mean / sample variance / stddev. Constrained with **SFINAE** (`enable_if_t<is_arithmetic_v<T>>`). Integer `T` accumulates in `double`. |
| `generic.h` | `sum(...)` (**variadic template + fold expression**), `clamp_val` (**SFINAE**), `zscore` (**`if constexpr`**), and the **STL** pipeline: `log_returns` (`std::transform`) and `rolling_stats_stl` (`std::vector`, `std::accumulate`, `std::sort` for the rolling median). |
| `main.cpp` | Unit tests (incl. `static_assert` compile-time checks) and the benchmark. |

## Build & run
```
g++ -std=c++17 -O2 -Wall -Wextra -o hw5 main.cpp && ./hw5
```
Compiles warning-free with g++ 13.3 (`-Wpedantic`) and clang++.

## Requirement → where
- **Generic container/algorithm:** `RingStat<T>`, `clamp_val<T>`, `zscore<T>`
- **STL rolling return stats:** `log_returns` + `rolling_stats_stl` (mean, sample sd, median)
- **Variadic template + fold:** `sum(Ts... xs)` → `(R{0} + ... + static_cast<R>(xs))`; result type is `std::common_type_t<Ts...>`, non-arithmetic args rejected by `static_assert`
- **SFINAE / `if constexpr`:** `RingStat` and `clamp_val` use `enable_if`; `zscore` uses `if constexpr` to promote integers to `double`

## Correctness
- `RingStat<double>` vs. full STL recomputation on 1,999 GBM log-returns, window 20: **max abs error 2.1e-17** over 1,980 windows.
- Compile-time checks: `sum(1,2,3)==6`, `sum(1,2.5)` is `double`, `zscore(int)` returns `double`, `zscore(float)` returns `float`.
- Negative tests (verified to fail compilation): `sum("a",1)`, `clamp_val(std::string...)`, `RingStat<std::string>`.

## Measurements
Data: 1,000,000 daily log-returns from a seeded geometric Brownian motion (μ=0.0002, σ=0.01).
Machine: Apple M2 (MacBook), Apple clang `-O2`, best of 5 runs (STL+median: 1 run).

| Window | RingStat (ms) | STL mean+sd (ms) | STL + median (ms) | RingStat speedup vs STL mean+sd |
|---:|---:|---:|---:|---:|
| 5 | 9.68 | 4.24 | 20.9 | 0.4× |
| 20 | 10.30 | 19.8 | 199 | 1.9× |
| 60 | 9.25 | 76.3 | 809 | 8.3× |
| 252 | 9.21 | 461 | 4,129 | 50.1× |

Full sample: annualized volatility 15.88% (σ·√252, target 15.87%).

## Discussion

**1. RingStat is O(1) per step, the STL recompute is O(w).**
RingStat stays at ~9.2–10.3 ms for every window size, about 9–10 ns per step. The STL recompute grows linearly with w: dividing time by (windows × w) gives a roughly constant cost of 0.8–1.8 ns per element touched. Going from w = 20 to w = 252 (12.6× larger) costs 23× more time, so the per-element cost also rises somewhat once the window no longer fits in L1.

**2. Why the naive loop wins at w = 5, and where the crossover is.**
RingStat does a fixed amount of work per step regardless of w: an evict-and-add update of `sum` and `sumsq`, three divisions (mean, plus two inside the variance), and one `sqrt`. The naive loop at w = 5 only adds 10 numbers per window. Its windows are also independent of each other, so the out-of-order CPU can overlap consecutive iterations. RingStat's steps form a serial chain through `sum_` and `sumsq_`, and each step must wait for the previous one.
Using about 9.5 ns per RingStat step and about 1 ns per element for the naive loop, the two break even around **w ≈ 10**. This matches the measurements: naive is faster at 5, RingStat at 20.
I first suspected the `%` wrap-around (`head_ = (head_ + 1) % cap_`). Replacing it with a compare-and-reset branch changed timings by less than run-to-run noise (tested on a separate x86 machine), so the modulo is not the bottleneck.

**3. The rolling median dominates.**
The median needs a copy plus `std::sort` per window, O(w log w). It is 5–11× slower than mean+sd alone. From w = 20 to 252 its time grows 20.7×, close to the 23× that w log w predicts. An incremental structure would bring it to O(log w) per step: two heaps with lazy deletion, or an order-statistic tree.

**4. Zero-overhead abstraction.**
All generic machinery is resolved at compile time: the `enable_if` constraints, the `static_assert` in `sum`, and the `if constexpr` branches in `zscore`. `RingStat<double>` therefore compiles to the same arithmetic you would hand-write for `double`. Misuse fails at compile time, not at run time. `RingStat<std::string>`, `clamp_val` on strings and `sum("a", 1)` are all rejected, as verified in the negative tests.

**5. Numerical caveat.**
The running sum-of-squares formula, var = (Σx² − (Σx)²/n)/(n−1), suffers catastrophic cancellation when |mean| ≫ sd. It would be inaccurate on raw prices (mean ≈ 100, sd ≈ 1). On log-returns (mean ≈ 0) it is accurate, as the 2.1e-17 error against the two-pass STL result shows. That is why the pipeline converts prices to returns first. For general inputs, Welford's online update would be the robust choice at similar cost. Variance is clamped at 0 to absorb tiny negative values from rounding.
