/**
 * simoncatbot-opencv RPP HAL - Core Operations (GPU + CPU unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED -> OpenCV native
 */

#include "rpp_hal_core.hpp"
#include "rpp_precomp.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>
#include <rpp/rppt_tensor_statistical_operations.h>
#include <rpp/rppt_tensor_color_augmentations.h>
#include <cmath>

using namespace cv::hal::rpp;

namespace {

// Build a full-image single-channel 8U descriptor + ROI (shared by bitwise ops).
inline void buildBitwise(RpptDesc& desc, RpptROI& roi, int width, int height) {
    buildRppDescNHWC(desc, width, height, 1, CV_8U);
    buildFullRoi(roi, width, height);
}

inline RppBuf buf8u1(const void* p, size_t step, int w, int h) {
    return RppBuf{ p, step, w, h, CV_8U, 1 };
}

} // namespace

// =========================================================================
// BITWISE (and / or / xor / not) — via runRpp executor
// =========================================================================

extern "C" int rpp_hal_and8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RpptDesc desc; RpptROI roi; buildBitwise(desc, roi, width, height);
    RppBuf srcs[2] = { buf8u1(src1_data, src1_step, width, height),
                       buf8u1(src2_data, src2_step, width, height) };
    RppBuf dst = buf8u1(dst_data, dst_step, width, height);
    return runRpp(srcs, 2, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_bitwise_and(s[0], s[1], &desc, d, &desc, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_or8u(const uchar* src1_data, size_t src1_step,
                 const uchar* src2_data, size_t src2_step,
                 uchar* dst_data, size_t dst_step,
                 int width, int height) {
    RpptDesc desc; RpptROI roi; buildBitwise(desc, roi, width, height);
    RppBuf srcs[2] = { buf8u1(src1_data, src1_step, width, height),
                       buf8u1(src2_data, src2_step, width, height) };
    RppBuf dst = buf8u1(dst_data, dst_step, width, height);
    return runRpp(srcs, 2, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_bitwise_or(s[0], s[1], &desc, d, &desc, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_xor8u(const uchar* src1_data, size_t src1_step,
                  const uchar* src2_data, size_t src2_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RpptDesc desc; RpptROI roi; buildBitwise(desc, roi, width, height);
    RppBuf srcs[2] = { buf8u1(src1_data, src1_step, width, height),
                       buf8u1(src2_data, src2_step, width, height) };
    RppBuf dst = buf8u1(dst_data, dst_step, width, height);
    return runRpp(srcs, 2, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_bitwise_xor(s[0], s[1], &desc, d, &desc, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_not8u(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height) {
    RpptDesc desc; RpptROI roi; buildBitwise(desc, roi, width, height);
    RppBuf srcs[1] = { buf8u1(src_data, src_step, width, height) };
    RppBuf dst = buf8u1(dst_data, dst_step, width, height);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_bitwise_not(s[0], &desc, d, &desc, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

// =========================================================================
// inRange — RPP threshold outputs a 255/0 in-range binary mask, which matches
// OpenCV inRange for single-channel input. (Multi-channel inRange ANDs the
// per-channel results into one mask; RPP thresholds each channel separately,
// so we only claim the cn==1 case and let native handle the rest.)
// =========================================================================

namespace {

inline int runInRange(const uchar* src_data, size_t src_step,
                      uchar* dst_data, size_t dst_step, int dst_depth,
                      int width, int height, int cn, int depth,
                      float lo, float hi) {
    if (cn != 1) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (dst_depth != CV_8U) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth != CV_8U && depth != CV_32F) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    // RPP HOST threshold deviates (like resize/warp); GPU path only.
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, 1, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, 1, depth);
    RpptROI roi; buildFullRoi(roi, width, height);
    Rpp32f minT = lo, maxT = hi;

    RppBuf srcs[1] = { RppBuf{ src_data, src_step, width, height, depth, 1 } };
    RppBuf dst = RppBuf{ dst_data, dst_step, width, height, depth, 1 };
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_threshold(s[0], &srcDesc, d, &dstDesc, &minT, &maxT,
                                  &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

} // namespace

extern "C" int rpp_hal_inRange8u(const uchar* src_data, size_t src_step,
                                 uchar* dst_data, size_t dst_step, int dst_depth,
                                 int width, int height, int cn,
                                 uchar lower_bound, uchar upper_bound) {
    return runInRange(src_data, src_step, dst_data, dst_step, dst_depth,
                      width, height, cn, CV_8U,
                      static_cast<float>(lower_bound), static_cast<float>(upper_bound));
}

extern "C" int rpp_hal_inRange32f(const uchar* src_data, size_t src_step,
                                  uchar* dst_data, size_t dst_step, int dst_depth,
                                  int width, int height, int cn,
                                  double lower_bound, double upper_bound) {
    return runInRange(src_data, src_step, dst_data, dst_step, dst_depth,
                      width, height, cn, CV_32F,
                      static_cast<float>(lower_bound), static_cast<float>(upper_bound));
}

// =========================================================================
// Math routines with no suitable RPP tensor equivalent -> fallback
// =========================================================================

extern "C" int rpp_hal_add8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_add16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul16s(const short*, size_t,
                   const short*, size_t,
                   short*, size_t,
                   int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_add32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_sub32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_mul32f(const float*, size_t,
                   const float*, size_t,
                   float*, size_t,
                   int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_div32f(const float* src1_data, size_t src1_step,
                   const float* src2_data, size_t src2_step,
                   float* dst_data, size_t dst_step,
                   int width, int height, double scale) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scale;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_recip32f(const float*, size_t,
                     float*, size_t,
                     int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_addWeighted8u(const uchar* src1_data, size_t src1_step,
                            const uchar* src2_data, size_t src2_step,
                            uchar* dst_data, size_t dst_step,
                            int width, int height,
                            const double scalars[3]) {
    (void)src1_data; (void)src1_step; (void)src2_data; (void)src2_step;
    (void)dst_data; (void)dst_step; (void)width; (void)height; (void)scalars;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_addWeighted32f(const float* src1_data, size_t src1_step,
                             const float* src2_data, size_t src2_step,
                             float* dst_data, size_t dst_step,
                             int width, int height,
                             const double scalars[3]) {
    // RPP blend computes alpha*src1 + (1-alpha)*src2. OpenCV addWeighted is
    // alpha*src1 + beta*src2 + gamma, so it only matches when beta == 1-alpha
    // and gamma == 0. GPU-only (RPP HOST color ops deviate like resize/warp).
    const double alpha = scalars[0], beta = scalars[1], gamma = scalars[2];
    if (std::fabs(beta - (1.0 - alpha)) > 1e-6) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (std::fabs(gamma) > 1e-6) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // width is already pre-multiplied by channels by the caller (continuous
    // arithmetic path), so treat as a single-channel F32 buffer.
    RpptDesc desc; buildRppDescNHWC(desc, width, height, 1, CV_32F);
    RpptROI roi; buildFullRoi(roi, width, height);
    Rpp32f alphaT = static_cast<Rpp32f>(alpha);

    RppBuf srcs[2] = { RppBuf{ src1_data, src1_step, width, height, CV_32F, 1 },
                       RppBuf{ src2_data, src2_step, width, height, CV_32F, 1 } };
    RppBuf dst = RppBuf{ dst_data, dst_step, width, height, CV_32F, 1 };
    return runRpp(srcs, 2, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_blend(s[0], s[1], &desc, d, &desc, &alphaT, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_cvt8u16s(const uchar*, size_t,
                     short*, size_t,
                     int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt16s8u(const short*, size_t,
                     uchar*, size_t,
                     int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt16s32f(const short*, size_t,
                      float*, size_t,
                      int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvt32f16s(const float*, size_t,
                      short*, size_t,
                      int, int, double) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs8u(const uchar*, size_t,
                  uchar*, size_t,
                  int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs16s(const short*, size_t,
                   short*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_abs32f(const float*, size_t,
                   float*, size_t,
                   int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp8u(const uchar*, size_t,
                  const uchar*, size_t,
                  uchar*, size_t,
                  int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp16s(const short*, size_t,
                   const short*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp32f(const float*, size_t,
                   const float*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cmp64f(const double*, size_t,
                   const double*, size_t,
                   uchar*, size_t,
                   int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx8u(const uchar*, size_t,
                          double*, double*, int*, int*,
                          int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx16s(const short*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx32f(const float*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_minMaxIdx64f(const double*, size_t,
                           double*, double*, int*, int*,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero8u(const uchar*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero16s(const short*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero32f(const float*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_countNonZero64f(const double*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct8u(const uchar*, size_t,
                         const uchar*, size_t,
                         int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct8s(const char*, size_t,
                         const char*, size_t,
                         int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct16u(const ushort*, size_t,
                          const ushort*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct32f(const float*, size_t,
                          const float*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_dotProduct64f(const double*, size_t,
                          const double*, size_t,
                          int, int, double*) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev8u(const uchar* src_data, size_t src_step,
                         int width, int height, double* meanVal, double* stdDevVal,
                         uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev16u(const ushort* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev32f(const float* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_meanStdDev64f(const double* src_data, size_t src_step,
                          int width, int height, double* meanVal, double* stdDevVal,
                          uchar* mask, size_t maskStep) {
    (void)src_data; (void)src_step; (void)width; (void)height;
    (void)meanVal; (void)stdDevVal; (void)mask; (void)maskStep;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral8u(const uchar*, size_t,
                     double*, size_t,
                     int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral32f(const float*, size_t,
                        double*, size_t,
                        int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_integral32s(const int*, size_t,
                        double*, size_t,
                        int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray8u(const uchar*, size_t,
                           uchar*, size_t,
                           int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray16u(const ushort*, size_t,
                            ushort*, size_t,
                            int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray32f(const float*, size_t,
                            float*, size_t,
                            int, int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtGraytoBGR8u(const uchar*, size_t,
                           uchar*, size_t,
                           int, int, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_lut(const uchar* src_data, size_t src_step, size_t src_type,
                const uchar* lut_data, size_t lut_channel_size, size_t lut_channels,
                uchar* dst_data, size_t dst_step, int width, int height) {
    // RPP lut applies a single 8U table to all channels. Match only the 8U,
    // single-table (lut_channels == 1), 8-bit-entry case; per-channel LUTs or
    // wider element types fall back to native. GPU-only (RPP HOST deviates).
    const int depth = CV_MAT_DEPTH(static_cast<int>(src_type));
    const int cn = CV_MAT_CN(static_cast<int>(src_type));
    if (depth != CV_8U) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (lut_channels != 1) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (lut_channel_size != 1) return CV_HAL_ERROR_NOT_IMPLEMENTED;  // 8-bit entries
    if (cn != 1 && cn != 3) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, cn, CV_8U);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, cn, CV_8U);
    RpptROI roi; buildFullRoi(roi, width, height);

    RppBuf srcs[1] = { RppBuf{ src_data, src_step, width, height, CV_8U, cn } };
    RppBuf dst = RppBuf{ dst_data, dst_step, width, height, CV_8U, cn };
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            // lutPtr lives in HOST/pinned memory for both backends.
            return rppt_lut(s[0], &srcDesc, d, &dstDesc,
                            const_cast<uchar*>(lut_data), &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_magnitude32f(const float*, const float*,
                         float*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_magnitude64f(const double*, const double*,
                         double*, int) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
