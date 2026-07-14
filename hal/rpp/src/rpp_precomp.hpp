/**
 * simoncatbot-opencv RPP HAL - Shared precompiled helpers
 *
 * Centralizes path selection, the min-size GPU guard, and the runRpp() executor
 * so individual ops carry no upload/handle/download/free boilerplate. This mirrors
 * the Intel IPP HAL layout (hal/ipp/src/precomp_ipp.hpp): converters + a common
 * execution harness shared by every op-family source file.
 */

#ifndef __RPP_PRECOMP_HPP__
#define __RPP_PRECOMP_HPP__

#include "rpp_hal_utils.hpp"
#include <cstring>
#include <cstdlib>
#include <functional>

#ifdef RPP_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace cv { namespace hal { namespace rpp {

// ---------------------------------------------------------------------------
// Path selection (GPU / CPU / native fallback) — shared by all op files
// ---------------------------------------------------------------------------

enum RppPath { RPP_NONE, RPP_GPU, RPP_CPU };

inline bool envTrue(const char* name) {
    const char* v = getenv(name);
    return v && (strcmp(v, "1") == 0 || strcmp(v, "yes") == 0 || strcmp(v, "true") == 0);
}

inline RppPath selectRppPath() {
    if (envTrue("OPENCV_RPP_DISABLE")) return RPP_NONE;
    if (envTrue("OPENCV_RPP_FORCE_CPU")) return isRppCpuAvailable() ? RPP_CPU : RPP_NONE;
    if (isRppGpuAvailable()) return RPP_GPU;
    if (isRppCpuAvailable()) return RPP_CPU;
    return RPP_NONE;
}

// Below this many destination pixels the per-call host<->device copy dominates
// and OpenCV native beats the GPU; fall back rather than pay the transfer.
// Overridable at runtime via OPENCV_RPP_MIN_GPU_PIXELS.
inline size_t minGpuPixels() {
    const char* v = getenv("OPENCV_RPP_MIN_GPU_PIXELS");
    if (v) { char* end = nullptr; long n = strtol(v, &end, 10); if (end != v && n >= 0) return (size_t)n; }
    return 64 * 64;
}

#ifdef RPP_BACKEND_HIP
inline void clearStickyHipError() { (void)hipGetLastError(); }
#else
inline void clearStickyHipError() {}
#endif

// ---------------------------------------------------------------------------
// runRpp executor
// ---------------------------------------------------------------------------

// Describes one host image buffer that the executor uploads (inputs) or
// downloads into (output).
struct RppBuf {
    const void* host;   // host pointer (src) / destination pointer (dst)
    size_t step;
    int w, h, depth, cn;
};

// The op callable receives resolved pointers that are DEVICE pointers on the GPU
// path and HOST pointers on the CPU path. It builds its own RpptDesc/ROI/params
// (dims are known at the call site) and invokes the specific rppt_* function.
//   srcDev[i] : resolved pointer for input i   (nSrc entries)
//   dstDev    : resolved pointer for the output
using RppOp = std::function<bool(void** srcDev, int nSrc, void* dstDev,
                                 rppHandle_t handle, RppBackend backend)>;

// Executes an RPP op end to end. Returns CV_HAL_ERROR_OK or NOT_IMPLEMENTED.
// - selects GPU/CPU/native path (respects env toggles + min-size guard),
// - uploads inputs / allocates the output on GPU (single-copy when contiguous),
// - reuses one persistent handle (no per-call rppCreate/rppDestroy),
// - downloads the output and returns buffers to the device pool.
int runRpp(const RppBuf* srcs, int nSrc, const RppBuf& dst, const RppOp& op);

}}} // namespace cv::hal::rpp

#endif // __RPP_PRECOMP_HPP__
