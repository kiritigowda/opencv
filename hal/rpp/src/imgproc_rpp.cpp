/**
 * simoncatbot-opencv RPP HAL - Imgproc Operations (Unified dispatch)
 *
 * Architecture:
 *   1. GPU: Uses RPP HIP backend with device memory upload/download
 *   2. CPU: Uses RPP HOST backend (no memory copies, direct pointer pass)
 *   3. Fallback: Returns CV_HAL_ERROR_NOT_IMPLEMENTED -> OpenCV native
 */

#include "rpp_hal_imgproc.hpp"
#include "rpp_precomp.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <rpp/rppt_tensor_filter_augmentations.h>

using namespace cv::hal::rpp;

namespace {

    inline RpptInterpolationType cvInterpolationToRpp(int interpolation) {
        switch (interpolation) {
            case 0: return NEAREST_NEIGHBOR; // INTER_NEAREST
            case 1: return BILINEAR;         // INTER_LINEAR
            case 2: return BICUBIC;          // INTER_CUBIC
            case 3: return LANCZOS;          // INTER_LANCZOS4 (closest)
            default: return BILINEAR;
        }
    }

    inline bool supportedDepth(int depth) {
        return depth == CV_8U || depth == CV_32F || depth == CV_8S;
    }

    inline RppBuf makeBuf(const void* p, size_t step, int w, int h, int depth, int cn) {
        return RppBuf{ p, step, w, h, depth, cn };
    }

}

// =========================================================================
// FLIP
// =========================================================================

extern "C" int rpp_hal_flip(int src_type,
                            const uchar* src_data, size_t src_step,
                            int src_width, int src_height,
                            uchar* dst_data, size_t dst_step,
                            int flip_mode) {
    int cn = CV_MAT_CN(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, src_width, src_height, cn, depth);
    RpptROI roi; buildFullRoi(roi, src_width, src_height);

    Rpp32u horizontal = (flip_mode == 1 || flip_mode == -1) ? 1 : 0;
    Rpp32u vertical   = (flip_mode == 0 || flip_mode == -1) ? 1 : 0;

    RppBuf srcs[1] = { makeBuf(src_data, src_step, src_width, src_height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, src_width, src_height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_flip(s[0], &srcDesc, d, &dstDesc, &horizontal, &vertical,
                             &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

// =========================================================================
// RESIZE — GPU only (RPP HOST backend deviates on bilinear resize)
// =========================================================================

extern "C" int rpp_hal_resize(int src_type,
                                const uchar* src_data, size_t src_step,
                                int src_width, int src_height,
                                uchar* dst_data, size_t dst_step,
                                int dst_width, int dst_height,
                                double inv_scale_x, double inv_scale_y,
                                int interpolation) {
    (void)inv_scale_x; (void)inv_scale_y;
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = CV_MAT_CN(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, dst_width, dst_height, cn, depth);
    RpptROI srcRoi; buildFullRoi(srcRoi, src_width, src_height);
    RpptImagePatch dstSize;
    dstSize.width = static_cast<Rpp32u>(dst_width);
    dstSize.height = static_cast<Rpp32u>(dst_height);
    RpptInterpolationType interp = cvInterpolationToRpp(interpolation);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, src_width, src_height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, dst_width, dst_height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_resize(s[0], &srcDesc, d, &dstDesc, &dstSize, interp,
                               &srcRoi, XYWH, h, be) == RPP_SUCCESS;
        });
}

// =========================================================================
// WARP AFFINE — GPU only (RPP HOST backend deviates on warp)
// =========================================================================

extern "C" int rpp_hal_warpAffine(int src_type,
                                    const uchar* src_data, size_t src_step,
                                    int src_width, int src_height,
                                    uchar* dst_data, size_t dst_step,
                                    int dst_width, int dst_height,
                                    const double M[6], int interpolation,
                                    int borderType, const double borderValue[4]) {
    (void)borderType; (void)borderValue;
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = CV_MAT_CN(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // OpenCV HAL passes the forward 2x3 affine matrix; RPP uses the same layout.
    // Per the RPP docs, affineTensor must live in pinned/HIP memory for the HIP
    // backend and HOST memory for the HOST backend, so the GPU path uploads it.
    float affine[6];
    for (int i = 0; i < 6; ++i) affine[i] = static_cast<float>(M[i]);

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, dst_width, dst_height, cn, depth);
    RpptROI srcRoi; buildFullRoi(srcRoi, src_width, src_height);
    RpptInterpolationType interp = cvInterpolationToRpp(interpolation);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, src_width, src_height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, dst_width, dst_height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) -> bool {
            Rpp32f* affinePtr = affine;
#ifdef RPP_BACKEND_HIP
            void* d_affine = nullptr;
            if (be == RPP_HIP_BACKEND) {
                if (hipMalloc(&d_affine, sizeof(affine)) != hipSuccess) return false;
                if (hipMemcpy(d_affine, affine, sizeof(affine), hipMemcpyHostToDevice) != hipSuccess) {
                    (void)hipFree(d_affine); return false;
                }
                affinePtr = static_cast<Rpp32f*>(d_affine);
            }
#endif
            bool ok = rppt_warp_affine(s[0], &srcDesc, d, &dstDesc, affinePtr, interp,
                                       &srcRoi, XYWH, h, be) == RPP_SUCCESS;
#ifdef RPP_BACKEND_HIP
            if (d_affine) (void)hipFree(d_affine);
#endif
            return ok;
        });
}

// =========================================================================
// BOX FILTER
// =========================================================================

extern "C" int rpp_hal_boxFilter(const uchar* src_data, size_t src_step,
                                 uchar* dst_data, size_t dst_step,
                                 int width, int height, int src_depth, int dst_depth, int cn,
                                 int margin_left, int margin_top, int margin_right, int margin_bottom,
                                 size_t ksize_width, size_t ksize_height,
                                 int anchor_x, int anchor_y,
                                 bool normalize, int border_type) {
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;
    (void)anchor_x; (void)anchor_y; (void)normalize;

    // RPP box_filter requires square kernel and only supports REPLICATE border.
    if (ksize_width != ksize_height) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (border_type != cv::BORDER_REPLICATE) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_depth != dst_depth) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!supportedDepth(src_depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (cn != 1 && cn != 3) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, cn, src_depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, cn, dst_depth);
    RpptROI roi; buildFullRoi(roi, width, height);
    Rpp32u kernelSize = static_cast<Rpp32u>(ksize_width);
    RpptImageBorderType border = REPLICATE;

    RppBuf srcs[1] = { makeBuf(src_data, src_step, width, height, src_depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, width, height, dst_depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_box_filter(s[0], &srcDesc, d, &dstDesc, kernelSize, border,
                                   &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_gaussianBlur(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int cn,
                                    size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                                    size_t ksize_width, size_t ksize_height,
                                    double sigmaX, double sigmaY, int border_type) {
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;

    // RPP gaussian_filter: square kernel (3/5/7/9 optimized), REPLICATE border,
    // c = 1/3, single stdDev per image. OpenCV uses separable sigmaX/sigmaY;
    // only the isotropic case maps.
    if (ksize_width != ksize_height) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (border_type != cv::BORDER_REPLICATE) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (cn != 1 && cn != 3) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const Rpp32u kernelSize = static_cast<Rpp32u>(ksize_width);
    if (kernelSize == 0 || (kernelSize % 2) == 0) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // OpenCV derives sigma from kernel size when sigma <= 0 (see getGaussianKernel):
    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8. RPP applies one stdDev for both axes,
    // so require sigmaX == sigmaY (or sigmaY unset) to preserve semantics.
    double sigma = sigmaX;
    if (sigma <= 0) sigma = 0.3 * ((static_cast<double>(kernelSize) - 1) * 0.5 - 1) + 0.8;
    if (sigmaY > 0 && sigmaY != sigmaX) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    Rpp32f stdDev = static_cast<Rpp32f>(sigma);

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, cn, depth);
    RpptROI roi; buildFullRoi(roi, width, height);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, width, height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, width, height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_gaussian_filter(s[0], &srcDesc, d, &dstDesc, &stdDev, kernelSize,
                                        REPLICATE, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

extern "C" int rpp_hal_medianBlur(const uchar* src_data, size_t src_step,
                                  uchar* dst_data, size_t dst_step,
                                  int width, int height, int depth, int cn, int ksize) {
    // RPP median_filter: square kernel (3/5/7/9 optimized), REPLICATE border, c = 1/3.
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (cn != 1 && cn != 3) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (ksize <= 0 || (ksize % 2) == 0) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    const Rpp32u kernelSize = static_cast<Rpp32u>(ksize);

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, cn, depth);
    RpptROI roi; buildFullRoi(roi, width, height);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, width, height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, width, height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_median_filter(s[0], &srcDesc, d, &dstDesc, kernelSize,
                                      REPLICATE, &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

// =========================================================================
// DERIVATIVES
// =========================================================================

extern "C" int rpp_hal_sobel(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int src_depth, int dst_depth, int cn,
                             int margin_left, int margin_top, int margin_right, int margin_bottom,
                             int dx, int dy, int ksize, double scale, double delta, int border_type) {
    (void)margin_left; (void)margin_top; (void)margin_right; (void)margin_bottom;

    // RPP sobel_filter: single-channel only (dst NCHW c=1), kernelSize 3/5/7,
    // sobelType 0=X, 1=Y, 2=XY. No scale/delta support, and RPP writes the same
    // dtype as input, so require src_depth == dst_depth and unit scale/zero delta.
    if (cn != 1) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (src_depth != dst_depth) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (!supportedDepth(src_depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (scale != 1.0 || delta != 0.0) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (border_type != cv::BORDER_REPLICATE) return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (ksize != 3 && ksize != 5 && ksize != 7) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Rpp32u sobelType;
    if (dx == 1 && dy == 0) sobelType = 0;
    else if (dx == 0 && dy == 1) sobelType = 1;
    else if (dx == 1 && dy == 1) sobelType = 2;
    else return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const Rpp32u kernelSize = static_cast<Rpp32u>(ksize);
    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, width, height, cn, src_depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, width, height, cn, dst_depth);
    RpptROI roi; buildFullRoi(roi, width, height);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, width, height, src_depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, width, height, dst_depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) {
            return rppt_sobel_filter(s[0], &srcDesc, d, &dstDesc, sobelType, kernelSize,
                                     &roi, XYWH, h, be) == RPP_SUCCESS;
        });
}

// =========================================================================
// FEATURES
// =========================================================================

extern "C" int rpp_hal_canny(const uchar* src_data, size_t src_step,
                             uchar* dst_data, size_t dst_step,
                             int width, int height, int cn,
                             double lowThreshold, double highThreshold, int ksize, bool L2gradient) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)cn;
    (void)lowThreshold; (void)highThreshold; (void)ksize; (void)L2gradient;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// =========================================================================
// GEOMETRY (remaining stubs)
// =========================================================================

extern "C" int rpp_hal_warpPerspective(int src_type,
                                       const uchar* src_data, size_t src_step,
                                       int src_width, int src_height,
                                       uchar* dst_data, size_t dst_step,
                                       int dst_width, int dst_height,
                                       const double M[9], int interpolation,
                                       int borderType, const double borderValue[4]) {
    (void)borderType; (void)borderValue;
    // GPU only (RPP HOST warp deviates, same as warpAffine).
    if (selectRppPath() != RPP_GPU) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int cn = CV_MAT_CN(src_type);
    int depth = CV_MAT_DEPTH(src_type);
    if (!supportedDepth(depth)) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    float persp[9];
    for (int i = 0; i < 9; ++i) persp[i] = static_cast<float>(M[i]);

    RpptDesc srcDesc; buildRppDescNHWC(srcDesc, src_width, src_height, cn, depth);
    RpptDesc dstDesc; buildRppDescNHWC(dstDesc, dst_width, dst_height, cn, depth);
    RpptROI srcRoi; buildFullRoi(srcRoi, src_width, src_height);
    RpptInterpolationType interp = cvInterpolationToRpp(interpolation);

    RppBuf srcs[1] = { makeBuf(src_data, src_step, src_width, src_height, depth, cn) };
    RppBuf dst = makeBuf(dst_data, dst_step, dst_width, dst_height, depth, cn);
    return runRpp(srcs, 1, dst,
        [&](void** s, int, void* d, rppHandle_t h, RppBackend be) -> bool {
            Rpp32f* perspPtr = persp;
#ifdef RPP_BACKEND_HIP
            void* d_persp = nullptr;
            if (be == RPP_HIP_BACKEND) {
                if (hipMalloc(&d_persp, sizeof(persp)) != hipSuccess) return false;
                if (hipMemcpy(d_persp, persp, sizeof(persp), hipMemcpyHostToDevice) != hipSuccess) {
                    (void)hipFree(d_persp); return false;
                }
                perspPtr = static_cast<Rpp32f*>(d_persp);
            }
#endif
            bool ok = rppt_warp_perspective(s[0], &srcDesc, d, &dstDesc, perspPtr, interp,
                                            &srcRoi, XYWH, h, be) == RPP_SUCCESS;
#ifdef RPP_BACKEND_HIP
            if (d_persp) (void)hipFree(d_persp);
#endif
            return ok;
        });
}

// =========================================================================
// COLOR CONVERSIONS
// =========================================================================

extern "C" int rpp_hal_cvtBGRtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, int dcn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)dcn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoGray(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth,
                                    int scn, bool swapBlue) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtGraytoBGR(const uchar* src_data, size_t src_step,
                                    uchar* dst_data, size_t dst_step,
                                    int width, int height, int depth, int dcn) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtBGRtoHSV(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int scn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)scn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

extern "C" int rpp_hal_cvtHSVtoBGR(const uchar* src_data, size_t src_step,
                                   uchar* dst_data, size_t dst_step,
                                   int width, int height, int depth,
                                   int dcn, bool swapBlue, bool isFullRange, bool isHSV) {
    (void)src_data; (void)src_step; (void)dst_data; (void)dst_step;
    (void)width; (void)height; (void)depth; (void)dcn; (void)swapBlue;
    (void)isFullRange; (void)isHSV;
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
