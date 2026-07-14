/*
 * simoncatbot-opencv RPP HAL - Imgproc Module
 *
 * AMD ROCm Performance Primitives (RPP) HAL for OpenCV 5.x imgproc
 */

#ifndef __RPP_HAL_IMGPROC_HPP__
#define __RPP_HAL_IMGPROC_HPP__

#include <opencv2/core/base.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// FILTERING
// =========================================================================

int rpp_hal_boxFilter(const uchar* src_data, size_t src_step,
                      uchar* dst_data, size_t dst_step,
                      int width, int height, int src_depth, int dst_depth, int cn,
                      int margin_left, int margin_top, int margin_right, int margin_bottom,
                      size_t ksize_width, size_t ksize_height,
                      int anchor_x, int anchor_y,
                      bool normalize, int border_type);

int rpp_hal_gaussianBlur(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth, int cn,
                         size_t margin_left, size_t margin_top, size_t margin_right, size_t margin_bottom,
                         size_t ksize_width, size_t ksize_height,
                         double sigmaX, double sigmaY, int border_type);

int rpp_hal_medianBlur(const uchar* src_data, size_t src_step,
                       uchar* dst_data, size_t dst_step,
                       int width, int height, int depth, int cn, int ksize);

// =========================================================================
// DERIVATIVES — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_sobel(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, int src_depth, int dst_depth, int cn,
                  int margin_left, int margin_top, int margin_right, int margin_bottom,
                  int dx, int dy, int ksize, double scale, double delta, int border_type);

// =========================================================================
// FEATURES — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_canny(const uchar* src_data, size_t src_step,
                  uchar* dst_data, size_t dst_step,
                  int width, int height, int cn,
                  double lowThreshold, double highThreshold, int ksize, bool L2gradient);

// =========================================================================
// MORPHOLOGY — erode / dilate via the stateless HAL entry point.
// RPP only supports a full square box kernel; all other cases (custom
// structuring element, off-center anchor, ROI/submatrix, iterations>1)
// return NOT_IMPLEMENTED and fall back to OpenCV native.
// =========================================================================

int rpp_hal_morph_stateless(int operation,
                            const uchar* src_data, size_t src_step, int src_type,
                            uchar* dst_data, size_t dst_step, int dst_type,
                            int width, int height,
                            int src_full_width, int src_full_height, int src_roi_x, int src_roi_y,
                            int dst_full_width, int dst_full_height, int dst_roi_x, int dst_roi_y,
                            const uchar* kernel_data, size_t kernel_step, int kernel_type,
                            int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                            int borderType, const double borderValue[4],
                            int iterations, bool allowSubmatrix, bool allowInplace);

// =========================================================================
// GEOMETRY
// =========================================================================

int rpp_hal_remap32f(int src_type,
                     const uchar* src_data, size_t src_step,
                     int src_width, int src_height,
                     uchar* dst_data, size_t dst_step,
                     int dst_width, int dst_height,
                     float* mapx, size_t mapx_step,
                     float* mapy, size_t mapy_step,
                     int interpolation, int border_type,
                     const double border_value[4]);

int rpp_hal_resize(int src_type,
                   const uchar* src_data, size_t src_step,
                   int src_width, int src_height,
                   uchar* dst_data, size_t dst_step,
                   int dst_width, int dst_height,
                   double inv_scale_x, double inv_scale_y, int interpolation);

int rpp_hal_warpAffine(int src_type,
                       const uchar* src_data, size_t src_step,
                       int src_width, int src_height,
                       uchar* dst_data, size_t dst_step,
                       int dst_width, int dst_height,
                       const double M[6], int interpolation,
                       int borderType, const double borderValue[4]);

int rpp_hal_warpPerspective(int src_type,
                            const uchar* src_data, size_t src_step,
                            int src_width, int src_height,
                            uchar* dst_data, size_t dst_step,
                            int dst_width, int dst_height,
                            const double M[9], int interpolation,
                            int borderType, const double borderValue[4]);

int rpp_hal_flip(int src_type,
                 const uchar* src_data, size_t src_step,
                 int src_width, int src_height,
                 uchar* dst_data, size_t dst_step,
                 int flip_mode);

// =========================================================================
// COLOR CONVERSIONS — no RPP GPU APIs, always NOT_IMPLEMENTED
// =========================================================================

int rpp_hal_cvtBGRtoBGR(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int scn, int dcn, bool swapBlue);

int rpp_hal_cvtBGRtoGray(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth,
                         int scn, bool swapBlue);

int rpp_hal_cvtGraytoBGR(const uchar* src_data, size_t src_step,
                         uchar* dst_data, size_t dst_step,
                         int width, int height, int depth, int dcn);

int rpp_hal_cvtBGRtoHSV(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int scn, bool swapBlue, bool isFullRange, bool isHSV);

int rpp_hal_cvtHSVtoBGR(const uchar* src_data, size_t src_step,
                        uchar* dst_data, size_t dst_step,
                        int width, int height, int depth,
                        int dcn, bool swapBlue, bool isFullRange, bool isHSV);

#ifdef __cplusplus
}
#endif

// =========================================================================
// Register HAL hooks with OpenCV
// =========================================================================

#undef cv_hal_boxFilter
#define cv_hal_boxFilter rpp_hal_boxFilter

#undef cv_hal_gaussianBlur
#define cv_hal_gaussianBlur rpp_hal_gaussianBlur

#undef cv_hal_medianBlur
#define cv_hal_medianBlur rpp_hal_medianBlur

#undef cv_hal_sobel
#define cv_hal_sobel rpp_hal_sobel

#undef cv_hal_canny
#define cv_hal_canny rpp_hal_canny

#undef cv_hal_resize
#define cv_hal_resize rpp_hal_resize

#undef cv_hal_warpAffine
#define cv_hal_warpAffine rpp_hal_warpAffine

#undef cv_hal_warpPerspective
#define cv_hal_warpPerspective rpp_hal_warpPerspective

#undef cv_hal_remap32f
#define cv_hal_remap32f rpp_hal_remap32f

#undef cv_hal_morph_stateless
#define cv_hal_morph_stateless rpp_hal_morph_stateless

#undef cv_hal_flip
#define cv_hal_flip rpp_hal_flip

#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR rpp_hal_cvtBGRtoBGR
#undef cv_hal_cvtBGRtoGray
#define cv_hal_cvtBGRtoGray rpp_hal_cvtBGRtoGray
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR rpp_hal_cvtGraytoBGR
#undef cv_hal_cvtBGRtoHSV
#define cv_hal_cvtBGRtoHSV rpp_hal_cvtBGRtoHSV
#undef cv_hal_cvtHSVtoBGR
#define cv_hal_cvtHSVtoBGR rpp_hal_cvtHSVtoBGR

#endif // __RPP_HAL_IMGPROC_HPP__
