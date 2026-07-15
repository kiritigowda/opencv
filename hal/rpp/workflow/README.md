# RPP HAL Workflow

Development and verification tooling for the AMD **RPP (ROCm Performance Primitives)**
HAL backend. Nothing here is part of the OpenCV build — these are standalone benchmarks
and correctness tests used to develop, validate, and measure the HAL.

The HAL sources themselves live one level up in `hal/rpp/{include,src}`.

## Minimum RPP version

**RPP 3.1.0** is the minimum supported version. The HAL depends on the RPP 3.x
**unified API**: a single RPP library exposes both the CPU (HOST) and GPU (HIP)
backends, selected at runtime via `rppCreate(..., RPP_HOST_BACKEND | RPP_HIP_BACKEND)`
and the `rppt_*` tensor operations. The build enforces this floor in
`cmake/OpenCVFindRPP.cmake`, the top-level `CMakeLists.txt`, and `hal/rpp/CMakeLists.txt`.

## Backend architecture

The HAL dispatches each `cv_hal_*` call in this order:

1. **GPU (HIP)** — RPP HIP backend on an AMD GPU. Selected when a HIP device is present
   (and usable — see the probe in `../src/rpp_utils.cpp`).
2. **CPU (HOST)** — RPP HOST backend (multithreaded CPU kernels) when no usable GPU.
3. **Fallback** — returns `CV_HAL_ERROR_NOT_IMPLEMENTED` so OpenCV runs its native kernel.

## Runtime toggles

These environment variables control dispatch at runtime (read in `../src`):

| Variable                | Effect                                                        |
|-------------------------|--------------------------------------------------------------|
| `OPENCV_RPP_DISABLE=1`  | Disable the RPP HAL entirely; every op falls back to native. |
| `OPENCV_RPP_FORCE_CPU=1`| Force the RPP HOST (CPU) backend even if a GPU is present.    |
| `OPENCV_RPP_FORCE_GPU=1`| Force the RPP HIP (GPU) backend selection/probe.             |

## Files

### Tests
- **`test_rpp_correctness.cpp`** — Compares RPP HAL output against OpenCV native
  (toggling `OPENCV_RPP_DISABLE`) for bitwise, resize, boxFilter, warpAffine, flip.
  Prints `[PASS]`/`[FAIL]` with max diff vs. tolerance.
- **`test_rpp_hal.cpp`** — Smoke test that exercises each hooked op and reports timing;
  sanity-checks a few results.
- **`test_imgproc_correctness.cpp`** — Imgproc-focused correctness comparison of RPP
  output vs. OpenCV native reference.
- **`test_minimal.cpp`** — Minimal single-op sanity check (bitwise on a small mat).

### Benchmarks
- **`benchmark_rpp.cpp`** — Focused HAL-vs-native benchmark for resize, warpAffine,
  flip, boxFilter, bitwise at HD and 4K.
- **`benchmark_rpp_full.cpp`** — Comprehensive HAL benchmark across all implemented ops
  and multiple resolutions (VGA → 4K). Args: `[warmup] [iters]`.
- **`benchmark_rpp_native.cpp`** — Raw RPP benchmark with **no OpenCV integration**;
  calls `rppt_*` directly on HOST or HIP. Uploads/downloads once per case to match HAL
  behavior. Args: `[warmup] [iters]`. Force GPU with `OPENCV_RPP_FORCE_GPU=1`.
- **`bench_phase1.cpp` / `bench_phase2.cpp` / `bench_phase34.cpp` / `bench_phase56.cpp`** —
  Per-phase HAL-vs-native benchmarks for the ops added in each roadmap phase.

### Result snapshots
- `benchmark_results_rpp_native_host_100iters.txt`
- `benchmark_results_rpp_native_host_100iters_with_resize.txt`
- `benchmark_results_cpu.txt` / `benchmark_results_hip.txt` / `benchmark_results_native.txt`

  Captured RPP HAL full-benchmark runs (CPU/HIP/native paths) kept for reference/comparison.

## Building and running

Requires an OpenCV install built with `-DWITH_RPP=ON` and a ROCm/RPP install (≥ 3.1.0).

```bash
# From this directory:
./build.sh                 # builds all tools here
./build.sh clean           # removes built binaries

# Override discovery if needed:
OpenCV_DIR=/path/to/opencv/install ROCM_PATH=/opt/rocm ./build.sh
```

Run examples:

```bash
./test_rpp_correctness                       # correctness vs native
./benchmark_rpp_full 20 1000                 # 20 warmups, 1000 iters (HAL)
OPENCV_RPP_FORCE_CPU=1 ./benchmark_rpp_full  # force RPP HOST path
OPENCV_RPP_FORCE_GPU=1 ./benchmark_rpp_native 20 100   # raw RPP HIP
```

Built binaries stay in this directory and are git-ignored (see the repo `.gitignore`).
