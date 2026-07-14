/*
 * Phase 1 op benchmark: gaussianBlur, medianBlur, sobel, warpPerspective.
 * Compare RPP HAL (default GPU) vs native (OPENCV_RPP_DISABLE=1) to decide
 * per the guard-fallback policy which ops the HAL should keep intercepting.
 */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
using namespace cv;
using namespace std;

static double ms(chrono::steady_clock::time_point a, chrono::steady_clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() / 1000.0;
}
template <class F> static double bench(F f, int it) {
    for (int i = 0; i < 10; ++i) f();
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) f();
    auto t2 = chrono::steady_clock::now();
    return ms(t1, t2) / it;
}

int main() {
    cout << "OpenCV " << CV_VERSION << "\n";
    for (auto sz : { Size(1920, 1080), Size(3840, 2160) }) {
        Mat c(sz, CV_8UC3), g(sz, CV_8UC1), d;
        randu(c, 0, 255); randu(g, 0, 255);
        double Mp[9] = {1, 0.05, 10, 0.02, 1, 20, 0.0001, 0.0002, 1};
        Mat M(3, 3, CV_64F, Mp);
        cout << "--- " << sz.width << "x" << sz.height << " ---\n";
        cout << "gauss3x3   " << bench([&]{ GaussianBlur(c, d, Size(3,3), 0, 0, BORDER_REPLICATE); }, 200) << " ms\n";
        cout << "median3x3  " << bench([&]{ medianBlur(c, d, 3); }, 100) << " ms\n";
        cout << "sobel3x3   " << bench([&]{ Sobel(g, d, CV_8U, 1, 0, 3, 1, 0, BORDER_REPLICATE); }, 200) << " ms\n";
        cout << "warpPersp  " << bench([&]{ warpPerspective(c, d, M, sz, INTER_LINEAR, BORDER_REPLICATE); }, 100) << " ms\n";
    }
    return 0;
}
