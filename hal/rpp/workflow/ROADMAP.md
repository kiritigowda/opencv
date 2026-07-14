# RPP HAL Roadmap

Plan to bring the AMD RPP HAL to full coverage of the OpenCV operations that map
one-to-one (or closely) onto RPP tensor primitives, and to make those mappings
actually *fast* by adopting the optimization strategies proven in the Intel IPP HAL.

- **Scope:** 50 target ops — **42 direct 1:1** + **8 partial/adapted**.
- **RPP version:** 3.1.0 minimum (verified against 3.1.2 on the AMD test host).
- **Backends:** RPP HIP (GPU) and RPP HOST (CPU), runtime-dispatched, native fallback.

---

## 1. Current state (honest baseline)

The HAL *declares* ~75 `cv_hal_*` hooks, but most are stubs. What actually calls an
RPP kernel today:

| Op | Path working | Notes |
|----|--------------|-------|
| bitwise_and / or / xor / not (8UC1) | GPU + CPU | correctness verified |
| resize (8U/32F) | **GPU only** | CPU path disabled — RPP HOST resize deviates |
| warpAffine (8U/32F) | **GPU only** | CPU path disabled — RPP HOST warp deviates |
| flip | GPU + CPU | correctness verified |
| boxFilter (square kernel, REPLICATE, 8U/32F, cn 1/3) | GPU + CPU | correctness verified |
| gaussianBlur (square, isotropic, REPLICATE, cn 1/3) | GPU + CPU | Phase 1 — correctness verified (tol 16; RPP coeffs differ) |
| medianBlur (square kernel, cn 1/3) | GPU + CPU | Phase 1 — correctness verified |
| sobel (cn 1, ksize 3/5/7, scale 1/delta 0, REPLICATE) | GPU + CPU | Phase 1 — correctness verified (tol 24) |
| warpPerspective (8U/32F) | **GPU only** | Phase 1 — correctness verified |
| erode (full box kernel 3/5/7/9, REPLICATE, cn 1/3) | GPU + CPU | Phase 2 — correctness exact (tol 0) |
| dilate (full box kernel 3/5/7/9, REPLICATE, cn 1/3) | GPU + CPU | Phase 2 — correctness exact (tol 0) |
| inRange (single-channel 8u/32f) | **GPU only** | Phase 2 — via rppt_threshold; RPP HOST deviates so GPU-guarded |
| remap (32f maps, NEAREST/BILINEAR, REPLICATE) | **GPU only** | Phase 2 — correctness verified |

### Phase 2 benchmark (RX 7900 XT, RPP GPU vs native AVX)
| op | HD native | HD RPP | 4K native | 4K RPP | verdict |
|----|-----------|--------|-----------|--------|---------|
| erode 3x3 | 0.186 | 0.192 | 0.599 | 0.614 | **tie** |
| dilate 3x3 | 0.180 | 0.191 | 0.605 | 0.620 | **tie** |
| inRange 1c | 0.068 | 0.081 | 0.264 | 0.272 | **tie** |
| remap 3c | 0.686 | 0.715 | 2.158 | 2.183 | **tie** |

ms/op. Same story as Phase 1 — all tie native (±~3%), none lose, all stay enabled. Still
single-op copy-bound; op-chaining remains the path to an actual GPU win.

### Phase 1 benchmark (RX 7900 XT, RPP GPU vs native AVX)
| op | HD native | HD RPP | 4K native | 4K RPP | verdict |
|----|-----------|--------|-----------|--------|---------|
| gaussianBlur 3x3 | 0.100 | 0.093 | 0.570 | 0.589 | **tie** |
| medianBlur 3x3 | 0.422 | 0.437 | 2.539 | 2.482 | **tie** |
| sobel 3x3 | 0.095 | 0.107 | 0.298 | 0.313 | **tie** |
| warpPerspective | 0.906 | 0.919 | 4.045 | 4.087 | **tie** |

ms/op. All four tie native (±~2%) — no op *loses*, so all stay enabled (guard-fallback
policy only disables ops that lose at all sizes; the Phase 0 min-size guard still routes
tiny images to native). No single-op GPU win yet — confirms the standing conclusion that
the real payoff needs op-*chaining* (data resident on device across ops), not one-shot HAL
calls that pay a PCIe round-trip each.

Everything else in `core_rpp.cpp` (add, sub, mul, div, addWeighted, cvt*, abs, cmp,
minMaxIdx, countNonZero, dotProduct, meanStdDev, integral, cvtColor, LUT, magnitude)
and in `imgproc_rpp.cpp` (gaussianBlur, medianBlur, warpPerspective, sobel, canny,
cvtColor*) is a **stub returning `CV_HAL_ERROR_NOT_IMPLEMENTED`**.

**Baseline real working set was 8 ops** (bitwise x4, resize, warpAffine, flip, boxFilter).
Phase 1 added 4 (gaussianBlur, medianBlur, sobel, warpPerspective) → 12. Phase 2 added 4
(erode, dilate, inRange, remap) → **16 working**. This roadmap turns the remaining mappable
ops into real implementations.

### Why it isn't fast yet
Every current call does: `hipMalloc → hipMemcpy H2D (row loop) → rppCreate handle →
kernel → hipMemcpy D2H (row loop) → free`. The per-call host↔device copy and handle
setup dominate; for single ops on a discrete GPU this loses to OpenCV's AVX kernels.
This is exactly the cost model the IPP HAL inverts (Section 3).

---

## 2. Coverage matrix — the 50 target functions

Legend for **Status**: ✅ real & verified · 🟡 hooked-but-stub · ⬜ not hooked yet.

### 2a. Direct 1:1 matches (42)

#### Arithmetic — 9  (`rppt_tensor_arithmetic_operations.h`)
| # | OpenCV HAL hook | RPP function | Status | Notes |
|---|---|---|---|---|
| 1 | cv_hal_add8u/16s/32f | rppt_tensor_add_tensor | 🟡 | dtype via RpptDesc |
| 2 | cv_hal_sub8u/16s/32f | rppt_tensor_subtract_tensor | 🟡 | |
| 3 | cv_hal_mul8u/16s/32f | rppt_tensor_multiply_tensor | 🟡 | scale = post-multiply_scalar |
| 4 | cv_hal_div32f | rppt_tensor_divide_tensor | 🟡 | |
| 5 | cv_hal_addWeighted8u/32f | rppt_blend (or fma_scalar) | 🟡 | alpha blend |
| 6 | (add scalar) cv_hal_add* w/ Scalar | rppt_add_scalar | ⬜ | |
| 7 | (sub scalar) | rppt_subtract_scalar | ⬜ | |
| 8 | cv_hal_magnitude32f/64f | rppt_magnitude | 🟡 | 2-input |
| 9 | cv_hal_log32f | rppt_log | ⬜ | |

#### Bitwise — 4  (`rppt_tensor_bitwise_operations.h`)
| # | OpenCV HAL hook | RPP function | Status |
|---|---|---|---|
| 10 | cv_hal_and8u | rppt_bitwise_and | ✅ |
| 11 | cv_hal_or8u | rppt_bitwise_or | ✅ |
| 12 | cv_hal_xor8u | rppt_bitwise_xor | ✅ |
| 13 | cv_hal_not8u | rppt_bitwise_not | ✅ |

#### Filter — 4  (`rppt_tensor_filter_augmentations.h`)
| # | OpenCV HAL hook | RPP function | Status | Notes |
|---|---|---|---|---|
| 14 | cv_hal_boxFilter | rppt_box_filter | ✅ | square kernel, REPLICATE only |
| 15 | cv_hal_gaussianBlur | rppt_gaussian_filter | 🟡 | map sigma→kernel |
| 16 | cv_hal_medianBlur | rppt_median_filter | 🟡 | square kernel |
| 17 | cv_hal_sobel | rppt_sobel_filter | 🟡 | dx/dy → sobel dir |

#### Geometric — 10  (`rppt_tensor_geometric_augmentations.h`)
| # | OpenCV HAL hook | RPP function | Status | Notes |
|---|---|---|---|---|
| 18 | cv_hal_resize | rppt_resize | ✅ | GPU only today |
| 19 | cv_hal_warpAffine | rppt_warp_affine | ✅ | GPU only today |
| 20 | cv_hal_warpPerspective | rppt_warp_perspective | 🟡 | 3x3 matrix |
| 21 | cv_hal_flip | rppt_flip | ✅ | |
| 22 | (rotate) | rppt_rotate | ⬜ | used by warpAffine special-case |
| 23 | cv_hal_remap32f/16s | rppt_remap | ⬜ | map1/map2 → RPP tables |
| 24 | (transpose) cv_hal_transpose2d | rppt_transpose | ⬜ | |
| 25 | cv_hal_phase32f | rppt_phase | ⬜ | 2-input angle |
| 26 | (hconcat/vconcat) | rppt_concat | ⬜ | |
| 27 | cv_hal_undistort / initUndistort | rppt_lens_correction | ⬜ | partial fit |

#### Statistical — 7  (`rppt_tensor_statistical_operations.h`)
| # | OpenCV HAL hook | RPP function | Status |
|---|---|---|---|
| 28 | cv_hal_minMaxIdx | rppt_tensor_min / rppt_tensor_max | 🟡 |
| 29 | (sum) | rppt_tensor_sum | ⬜ |
| 30 | cv_hal_meanStdDev (mean) | rppt_tensor_mean | 🟡 |
| 31 | cv_hal_meanStdDev (stddev) | rppt_tensor_stddev | 🟡 |
| 32 | cv_hal_threshold | rppt_threshold | ⬜ |
| 33 | cv_hal_normalize | rppt_normalize | ⬜ |
| 34 | (countNonZero via threshold+sum) | rppt_threshold + rppt_tensor_sum | 🟡 |

#### Color / Data-exchange — 6
| # | OpenCV HAL hook | RPP function | Status | Notes |
|---|---|---|---|---|
| 35 | cv_hal_lut | rppt_lut | 🟡 | |
| 36 | cv_hal_cvtBGRtoGray | rppt_color_to_greyscale | 🟡 | |
| 37 | cv_hal_cvtColor YUV↔RGB | rppt_yuv_to_rgb | ⬜ | |
| 38 | (equalizeHist) | rppt_histogram_equalize | ⬜ | |
| 39 | cv_hal_addWeighted (alt) | rppt_blend | ⬜ | see #5 |
| 40 | (copyTo/convertTo copy) | rppt_copy | ⬜ | |

#### Morphological — 2  (`rppt_tensor_morphological_operations.h`)
| # | OpenCV HAL hook | RPP function | Status | Notes |
|---|---|---|---|---|
| 41 | cv_hal_morph (erode) | rppt_erode | ⬜ | box kernel only |
| 42 | cv_hal_morph (dilate) | rppt_dilate | ⬜ | box kernel only |

### 2b. Partial / adapted matches (8)

| # | OpenCV op | RPP function | Adaptation needed |
|---|---|---|---|
| 43 | convertScaleAbs / brightness | rppt_brightness | alpha/beta → RPP brightness+contrast params |
| 44 | contrast stretch | rppt_contrast | |
| 45 | Mat ROI copy / crop | rppt_crop / rppt_slice | ROI offset mapping |
| 46 | fisheye::undistortImage | rppt_fisheye | model differs; approximate |
| 47 | mixChannels / split-merge | rppt_channel_permute | layout mapping |
| 48 | LUT-based gamma | rppt_gamma_correction | gamma → LUT equivalence |
| 49 | convertTo (scale+shift) | rppt_fused_multiply_add_scalar | a*x+b |
| 50 | cvtColor BGR↔RGB (channel swap) | rppt_channel_permute | swapBlue |

---

## 3. Optimization strategy (adopted from the IPP HAL)

The IPP HAL (`hal/ipp/`) inverts the cost model: **expensive setup is done once and
amortized; per-call execution is cheap.** The transferable patterns:

### S1 — Separate init from execution; cache the spec/handle
IPP splits `GetSize → InitAlloc (once) → Run (per tile)` and stores the spec as a
member reused across calls (`resize_ipp.cpp`, `warp_ipp.cpp`).
**RPP action:** the HAL already reuses one `thread_local` GPU handle (good). Extend this
to a small **spec/param cache keyed by (op, dtype, cn, ksize, interp)** so repeated calls
(video frames) skip re-derivation. Never call `rppCreate`/`rppDestroy` per op.

### S2 — Kill the per-call host↔device copy (the #1 cost)
This is the single biggest win and has no IPP analogue (IPP is CPU-only, zero-copy).
**RPP actions:**
- **Device buffer pool** (already scaffolded in `rpp_utils.cpp` `DeviceBufferPool`) —
  make `freeHipPtr` size-aware so buffers are truly reused, not leaked/mis-binned.
- **Pinned host staging buffers** for faster, async H2D/D2H.
- **Contiguous fast path:** when `step == width*cn*elemSize`, do a single `hipMemcpy`
  instead of the current per-row loop (`uploadRawToHip`/`downloadRawFromHip`).
- **HIP streams + async copy** so upload/compute/download overlap.
- **CPU path is already zero-copy** (passes host pointers straight to RPP HOST) — prefer
  it for small images where PCIe copy dominates.

### S3 — Size/format guard matrix (fail fast before paying copy cost)
IPP uses `impl[depth][channels][interp]` tables and rejects unsupported combos immediately
(`warp_ipp.cpp:92`). **RPP action:** build a static support matrix per op; return
`NOT_IMPLEMENTED` *before* any hipMalloc when dtype/cn/border/kernel is unsupported.
Add a **min-size threshold**: below ~N pixels, GPU copy never wins → fall back to native
(mirrors IPP's `min_payload` heuristic in `warp_ipp.cpp:164`).

### S4 — Dispatch tables over boilerplate
Current code copy-pastes the full upload/handle/download/free sequence in every function
(~60 lines each). **RPP action:** factor a single templated executor,
e.g. `runRppUnary(op_fn, src, dst, desc...)` / `runRppBinary(...)`, and drive dtype×cn
selection from a table (IPP `warp_ipp.cpp:427`). Cuts ~1000 lines and centralizes the
copy/stream logic so S2 improvements apply everywhere at once.

### S5 — Threading / batching for the CPU (HOST) path
IPP tiles rows via `cv::parallel_for_` sized to cache (`precomp_ipp.hpp:72`).
**RPP action:** RPP HOST is internally multithreaded, but expose `numThreads` via the
handle and, where OpenCV batches (e.g. per-plane), use RPP's native **batch (n>1) tensor**
support to process planes/tiles in one call instead of N calls.

### S6 — File organization mirrors IPP
Split the monolithic `core_rpp.cpp`/`imgproc_rpp.cpp` into op-family files
(`arithmetic_rpp.cpp`, `filter_rpp.cpp`, `geometric_rpp.cpp`, `statistical_rpp.cpp`,
`color_rpp.cpp`, `morph_rpp.cpp`) + a shared `rpp_precomp.hpp` for enum/dtype converters
and the executor helpers. Matches `hal/ipp/src/*_ipp.cpp` layout.

---

## 4. Phased execution plan

**Phase 0 — Optimization foundation (do first; benefits every op).**
- S4 executor helpers (`runRppUnary/Binary/Filter/Geometric`) + S6 file split.
- S2 memory: size-aware pool, contiguous single-copy fast path, pinned buffers, streams.
- S3 guard matrix + min-size threshold.
- Verify: existing 8 ops still pass `test_rpp_correctness`; re-run `benchmark_rpp`
  to confirm the copy path improved (goal: GPU ≥ native at 4K, no worse at HD).

**Phase 1 — Complete the arithmetic/bitwise/statistical core (real, not stubs).**
- Fill stubs #1–9, #28–34 through the new executor. Add dtype coverage (16s/32f).
- Fix the CPU resize/warp deviations (#18/#19) or keep GPU-only with a documented guard.

**Phase 2 — Filters + geometry.**
- #15 gaussianBlur, #16 medianBlur, #17 sobel; #20 warpPerspective, #22 rotate,
  #23 remap, #24 transpose, #25 phase.

**Phase 3 — Color, morphology, partials.**
- #35–42 color/data-exchange + erode/dilate; then the 8 partials #43–50.

**Checkpoint after each phase:** all implemented ops pass correctness on GPU + CPU +
disabled paths; benchmark table updated in this folder.

---

## 5. Explicitly out of scope (RPP-only, no OpenCV 1:1)
The entire `effects` category (fog, rain, snow, noise, dropout, glitch, pixelate,
vignette, spatter, …) and ML-augmentation composites (crop_mirror_normalize, ricap,
resize_crop_mirror, color_jitter/twist, jpeg_compression_distortion). ~43 functions with
no OpenCV counterpart — not part of this HAL.

---

## 6. Success criteria
- All 50 ops implemented as real RPP calls (no silent stubs claiming support).
- Each passes `test_rpp_correctness`-style check vs native on GPU + CPU (or a documented,
  guarded fallback where an RPP backend is known-incorrect).
- On a discrete AMD GPU, the accelerated ops **meet or beat** OpenCV native AVX at
  ≥ FHD for compute-heavy ops (filters, warp, resize, cvtColor); small/cheap ops fall
  back to native below the size threshold rather than losing to copy overhead.
