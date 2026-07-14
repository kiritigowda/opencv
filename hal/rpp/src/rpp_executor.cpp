/**
 * simoncatbot-opencv RPP HAL - runRpp executor implementation
 *
 * Single place that owns the GPU upload/run/download/free lifecycle and the CPU
 * zero-copy path, so every op file stays a thin descriptor + one rppt_* call.
 */

#include "rpp_precomp.hpp"

namespace cv { namespace hal { namespace rpp {

namespace {

// Output-only buffers do not need their prior contents; inputs must be uploaded.
constexpr int kMaxSrc = 3;

} // namespace

int runRpp(const RppBuf* srcs, int nSrc, const RppBuf& dst, const RppOp& op) {
    if (nSrc < 0 || nSrc > kMaxSrc) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RppPath path = selectRppPath();
    if (path == RPP_NONE) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // Small-image guard: on GPU the transfer dominates; let native handle it.
    if (path == RPP_GPU) {
        const size_t dstPixels = static_cast<size_t>(dst.w) * static_cast<size_t>(dst.h);
        if (dstPixels < minGpuPixels()) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    RppBackend backend = (path == RPP_GPU) ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

#ifdef RPP_BACKEND_HIP
    if (path == RPP_GPU) {
        void* devSrc[kMaxSrc] = { nullptr, nullptr, nullptr };
        size_t srcBytes[kMaxSrc] = { 0, 0, 0 };
        void* devDst = nullptr;
        const size_t dstBytes = rawImageBytes(dst.w, dst.h, dst.depth, dst.cn);

        bool ok = true;
        for (int i = 0; i < nSrc && ok; ++i) {
            srcBytes[i] = rawImageBytes(srcs[i].w, srcs[i].h, srcs[i].depth, srcs[i].cn);
            ok = uploadRawToHip(srcs[i].host, srcs[i].step, srcs[i].w, srcs[i].h,
                                srcs[i].depth, srcs[i].cn, &devSrc[i]);
        }
        if (ok) {
            devDst = allocHipBuffer(dstBytes);   // output: no upload needed
            ok = (devDst != nullptr);
        }

        rppHandle_t handle = nullptr;
        if (ok) {
            handle = createRppGpuHandle(1);      // persistent, thread-local
            ok = (handle != nullptr);
        }

        if (ok) {
            ok = op(devSrc, nSrc, devDst, handle, backend);
            clearStickyHipError();
        }
        if (ok) {
            ok = downloadRawFromHip(devDst, const_cast<void*>(dst.host), dst.step,
                                    dst.w, dst.h, dst.depth, dst.cn);
        }

        for (int i = 0; i < nSrc; ++i) freeHipPtr(devSrc[i], srcBytes[i]);
        freeHipPtr(devDst, dstBytes);
        return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
#endif

    // CPU (HOST) path: RPP reads/writes host memory directly — zero copy.
    rppHandle_t handle = createRppCpuHandle(1);
    if (!handle) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    void* hostSrc[kMaxSrc] = { nullptr, nullptr, nullptr };
    for (int i = 0; i < nSrc; ++i) hostSrc[i] = const_cast<void*>(srcs[i].host);

    bool ok = op(hostSrc, nSrc, const_cast<void*>(dst.host), handle, backend);
    destroyRppCpuHandle(handle);
    return ok ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

}}} // namespace cv::hal::rpp
