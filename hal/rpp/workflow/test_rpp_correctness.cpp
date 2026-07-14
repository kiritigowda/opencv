/*
 * Correctness test for RPP HAL vs OpenCV native.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

static bool matEqual(const Mat& a, const Mat& b, int depth, double tol) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    Mat diff;
    absdiff(a, b, diff);
    if (depth == CV_8U) {
        double m = 0; minMaxLoc(diff, nullptr, &m);
        return m <= tol;
    } else if (depth == CV_32F) {
        double m = 0; minMaxLoc(diff, nullptr, &m);
        return m <= tol;
    }
    return false;
}

static void test(const string& name, const Mat& rppOut, const Mat& nativeOut, int depth, double tol) {
    bool ok = matEqual(rppOut, nativeOut, depth, tol);
    cout << (ok ? "[PASS] " : "[FAIL] ") << name;
    if (!ok) {
        double mn = 0, mx = 0;
        Mat diff;
        absdiff(rppOut, nativeOut, diff);
        minMaxLoc(diff, &mn, &mx);
        cout << " (max diff " << mx << " above tol " << tol << ")";
    }
    cout << endl;
}

int main() {
    cout << "=== RPP HAL Correctness Test ===" << endl;

    const int W = 640, H = 480;
    RNG rng(42);
    Mat gray8u(H, W, CV_8UC1);
    rng.fill(gray8u, RNG::UNIFORM, 0, 255);
    Mat gray8u2(H, W, CV_8UC1);
    rng.fill(gray8u2, RNG::UNIFORM, 0, 255);
    Mat color8u(H, W, CV_8UC3);
    rng.fill(color8u, RNG::UNIFORM, 0, 255);

    // Bitwise AND
    {
        Mat rppOut, nativeOut;
        bitwise_and(gray8u, gray8u2, rppOut);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        bitwise_and(gray8u, gray8u2, nativeOut);
        unsetenv("OPENCV_RPP_DISABLE");
        test("bitwise_and 8UC1", rppOut, nativeOut, CV_8U, 0);
    }

    // Bitwise NOT
    {
        Mat rppOut, nativeOut;
        bitwise_not(gray8u, rppOut);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        bitwise_not(gray8u, nativeOut);
        unsetenv("OPENCV_RPP_DISABLE");
        test("bitwise_not 8UC1", rppOut, nativeOut, CV_8U, 0);
    }

    // Resize 8UC1
    {
        Mat rppOut, nativeOut;
        resize(gray8u, rppOut, Size(W/2, H/2), 0, 0, INTER_LINEAR);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        resize(gray8u, nativeOut, Size(W/2, H/2), 0, 0, INTER_LINEAR);
        unsetenv("OPENCV_RPP_DISABLE");
        test("resize 8UC1 linear", rppOut, nativeOut, CV_8U, 2);
    }

    // Resize 8UC3
    {
        Mat rppOut, nativeOut;
        resize(color8u, rppOut, Size(W/2, H/2), 0, 0, INTER_LINEAR);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        resize(color8u, nativeOut, Size(W/2, H/2), 0, 0, INTER_LINEAR);
        unsetenv("OPENCV_RPP_DISABLE");
        test("resize 8UC3 linear", rppOut, nativeOut, CV_8U, 2);
    }

    // Box filter
    {
        Mat rppOut, nativeOut;
        boxFilter(color8u, rppOut, -1, Size(3,3), Point(-1,-1), true, BORDER_REPLICATE);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        boxFilter(color8u, nativeOut, -1, Size(3,3), Point(-1,-1), true, BORDER_REPLICATE);
        unsetenv("OPENCV_RPP_DISABLE");
        test("boxFilter 3x3 8UC3", rppOut, nativeOut, CV_8U, 2);
    }

    // Warp affine 8UC1
    {
        double M_data[6] = {1.0, 0.05, 10.0, 0.02, 1.0, 20.0};
        Mat M(2, 3, CV_64FC1, M_data);
        Mat rppOut, nativeOut;
        warpAffine(gray8u, rppOut, M, gray8u.size(), INTER_LINEAR, BORDER_REPLICATE);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        warpAffine(gray8u, nativeOut, M, gray8u.size(), INTER_LINEAR, BORDER_REPLICATE);
        unsetenv("OPENCV_RPP_DISABLE");
        test("warpAffine 8UC1", rppOut, nativeOut, CV_8U, 3);
    }

    // Warp affine 8UC3
    {
        double M_data[6] = {1.0, 0.05, 10.0, 0.02, 1.0, 20.0};
        Mat M(2, 3, CV_64FC1, M_data);
        Mat rppOut, nativeOut;
        warpAffine(color8u, rppOut, M, color8u.size(), INTER_LINEAR, BORDER_REPLICATE);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        warpAffine(color8u, nativeOut, M, color8u.size(), INTER_LINEAR, BORDER_REPLICATE);
        unsetenv("OPENCV_RPP_DISABLE");
        test("warpAffine 8UC3", rppOut, nativeOut, CV_8U, 3);
    }

    // Flip
    {
        Mat rppOut, nativeOut;
        flip(color8u, rppOut, 1);
        setenv("OPENCV_RPP_DISABLE", "1", 1);
        flip(color8u, nativeOut, 1);
        unsetenv("OPENCV_RPP_DISABLE");
        test("flip horizontal 8UC3", rppOut, nativeOut, CV_8U, 0);
    }

    cout << "=== Done ===" << endl;
    return 0;
}
