# RPP HAL — Project Summary & Benchmark Deep-Dive

_AMD RPP acceleration backend for OpenCV 5.x. Branch `rpp-hal-backend`._
_Benchmarks: santiago — Radeon RX 7900 XT + Ryzen 9 7900X, ROCm 7.14, RPP 3.1.2,
OpenCV 5.1.0-dev, 1920×1080, 100 iters / 5 warmups, 2026-07-15._

---

## 1. What was built

An OpenCV HAL backend that transparently routes `cv::` calls to AMD's ROCm
Performance Primitives (RPP), using the RPP 3.x **unified API** (one library
exposes both a HOST/CPU and a HIP/GPU backend). Runtime dispatch picks GPU → CPU →
native fallback, controlled by env toggles (`OPENCV_RPP_DISABLE`,
`OPENCV_RPP_FORCE_CPU`, `OPENCV_RPP_FORCE_GPU`).

**23 operations fully integrated and correctness-verified** (25/25 test cases pass
on GPU + CPU), built over 7 phases:

| Phase | Ops added | Count |
|-------|-----------|-------|
| Baseline | bitwise ×4, resize, warpAffine, flip, boxFilter | 8 |
| 0 | (no new ops) executor + memory-copy optimization foundation | 8 |
| 1 | gaussianBlur, medianBlur, sobel, warpPerspective | 12 |
| 2 | erode, dilate, inRange, remap | 16 |
| 3/4 | lut, equalizeHist, cvtColor BGR↔RGB, addWeighted | 20 |
| 5/6 | sum, meanStdDev, magnitude | 23 |

Architecture highlights:
- `runRpp` executor centralizes the GPU upload → alloc-output → run → download →
  pooled-free lifecycle and the CPU zero-copy path (each op is a thin descriptor +
  one `rppt_*` call).
- `runRppReduce` second executor for image→small-array reductions.
- Memory optimizations: single contiguous `hipMemcpy` fast path, size-aware device
  buffer pool, no pointless destination upload, `OPENCV_RPP_MIN_GPU_PIXELS` guard.
- Persistent thread-local RPP handle (no per-call `rppCreate`/`rppDestroy`).

---

## 2. Benchmark matrix (ms/op, lower is better)

Four measurable configs. (Direct RPP-HIP standalone timing is not reliable — see §4.)

| op | native | hal-cpu | hal-hip | rpp-cpu(pure) |
|----|-------:|--------:|--------:|--------------:|
| bitwise_and 8UC1 | 0.047 | 0.091 | 0.062 | 0.059 |
| bitwise_or 8UC1 | 0.047 | 0.084 | 0.056 | 0.059 |
| bitwise_xor 8UC1 | 0.048 | 0.080 | 0.056 | 0.058 |
| bitwise_not 8UC1 | 0.032 | 0.067 | 0.041 | 0.035 |
| boxFilter 3x3 8UC3 | 0.606 | 0.688 | 0.605 | 0.644 |
| gaussianBlur 3x3 | 0.100 | 0.095 | 0.095 | 2.004 |
| medianBlur 3x3 | 0.465 | 0.451 | 0.475 | 0.442 |
| sobel dx 3x3 8UC1 | 0.103 | 0.152 | 0.114 | ERR |
| resize down2x 8UC3 | 0.052 | 0.075 | 0.058 | 0.651 |
| warpAffine 8UC3 | 0.762 | 0.737 | 0.775 | 10.78 |
| warpPerspect 8UC3 | 0.935 | 0.915 | 0.930 | 10.69 |
| flip horiz 8UC3 | 1.967 | 0.590 | 1.978 | 0.538 |
| erode 3x3 8UC3 | 0.180 | 0.234 | 0.192 | ERR |
| dilate 3x3 8UC3 | 0.178 | 0.236 | 0.199 | ERR |
| remap 8UC3 bilin | 0.719 | 0.701 | 0.709 | 11.30 |
| inRange 8UC1 | 0.076 | 0.088 | 0.080 | n/a* |
| lut 8UC3 | 0.183 | 0.207 | 0.182 | 1.614 |
| equalizeHist 8UC1 | 0.275 | 0.305 | 0.289 | n/a* |
| cvtColor BGR2RGB | 0.131 | 0.153 | 0.136 | 0.250 |
| addWeighted 32f | 0.210 | 0.240 | 0.228 | 0.231 |
| sum 8UC1 | 0.049 | 0.068 | 0.056 | n/a* |
| meanStdDev 8UC1 | 0.043 | 0.062 | 0.051 | n/a* |
| magnitude 32f | 0.349 | 0.373 | 0.358 | 0.262 |

`ERR` = no RPP HOST implementation (sobel/erode/dilate are GPU-only in RPP 3.1.2).
`n/a*` = RPP HOST returned success in sub-µs without doing work — unreliable, which is
exactly why the HAL guards these ops to GPU-only.

---

## 3. Deep-dive findings

### 3.1 Single-op acceleration: essentially parity, no net win
- **Mean hal-hip / native ratio: 1.08×** (8% slower on average).
- **Mean hal-cpu / native ratio: 1.26×** (26% slower on average).
- hal-hip beats native on only **1/23** ops (gaussianBlur, 0.95×); it ties (±5%) on
  most and is modestly slower on the cheapest ops.

**Why:** every HAL call pays a full host→device→host round-trip over PCIe. For a
2 MP image that transfer (~6–12 MB round-trip) costs more than OpenCV's AVX kernel
takes to compute the whole operation. The GPU kernel itself is fast; the copy is the
tax. This is fundamental to a per-op HAL on a discrete GPU and matches every
per-phase result we recorded.

### 3.2 The cheaper the op, the worse the GPU looks
Ratios track compute intensity almost perfectly:
- Trivial ops (bitwise, ~0.03–0.05 ms native) → hal-hip **1.2–1.3× slower**: pure
  copy overhead, no compute to amortize it.
- Heavy ops (boxFilter, warp, remap, gaussianBlur, ~0.6–0.9 ms native) → **≈1.0×**:
  the copy is hidden behind real compute.
- The min-size guard (`OPENCV_RPP_MIN_GPU_PIXELS`) exists precisely so tiny images
  skip the GPU and avoid the losing trade.

### 3.3 The flip anomaly (native's weakness, not RPP's strength)
`flip` native = 1.97 ms but hal-cpu (RPP HOST) = 0.59 ms — a **3.3× win for RPP CPU**.
OpenCV's native horizontal flip on interleaved 8UC3 is comparatively slow; RPP's HOST
kernel is much better. hal-hip stays at native speed (1.98 ms) because the current
dispatch prefers GPU and eats the copy. **Actionable:** route `flip` (and similar
memory-bound reshapes) to the RPP HOST backend even when a GPU is present.

### 3.4 RPP HOST is a mixed bag
- Wins vs native: flip (3.3×), gaussianBlur, medianBlur, warpAffine (small).
- Losses: everything cheap (bitwise 1.7–2.1× slower), and it **has no HOST kernel**
  for sobel/erode/dilate or a reliable one for the reductions/threshold.
- Direct rpp-cpu "pure" numbers expose RPP HOST weak spots dramatically: warpAffine
  10.8 ms, remap 11.3 ms, resize 0.65 ms, lut 1.6 ms, gaussianBlur 2.0 ms — all far
  worse than both native and RPP-HIP. RPP's HOST path is not broadly competitive.

### 3.5 Correctness is solid everywhere it runs
All 25 correctness cases pass on GPU and CPU. Ops where RPP HOST deviates
(resize, warp, inRange, threshold-based, reductions, color) are guarded to GPU-only
and fall back to native on CPU — verified, no silent wrong answers.

---

## 4. Measurement caveat: direct RPP-HIP timing

Standalone direct-`rppt_*` timing on the HIP backend is **not reliably measurable**
and is omitted. RPP 3.x HIP kernels enqueue asynchronously on RPP's internal stream;
`hipDeviceSynchronize()` on the default stream returns in ~0.5 µs before the kernel
executes, and the first launch leaves a sticky context error that poisons subsequent
calls. The HAL works around this via its specific handle lifecycle + per-call error
clearing, and its GPU path is proven correct by the test suite — so **hal-hip is the
authoritative GPU measurement**, and it already includes the (dominant) copy cost.

---

## 5. Conclusions & where the real win is

1. **Coverage is broad and correct** — 23 ops, all verified, clean guard-and-fallback.
2. **Per-op single-image acceleration is not a win on a discrete GPU** — the PCIe
   round-trip dominates. hal-hip ≈ native; hal-cpu is generally slower except flip.
3. **The one clear, shippable single-op win is routing memory-bound ops (flip) to
   RPP HOST**, which is 3.3× native and needs no GPU transfer.
4. **The real GPU payoff requires op-chaining** — keeping data resident on-device
   across a *sequence* of ops so the H↔D copy is paid once for many kernels, not once
   per kernel. The executor already centralizes the copy boundary, so this is the
   natural next investment. OpenCV's one-op-at-a-time HAL dispatch does not trigger it
   today; it would need a batching/graph entry point or a UMat-backed path.

Recommended next steps, in priority order:
- Prototype an on-device op-chain (e.g. resize→boxFilter→cvtColor) to quantify the
  copy-amortization win.
- Add a HOST-preferred route for memory-bound ops where RPP HOST beats native (flip).
- Clean up the ~50 remaining dead `cv_hal_*` stub mappings so the HAL only advertises
  what it implements.
