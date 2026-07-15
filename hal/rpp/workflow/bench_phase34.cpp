/*
 * Phase 3/4 op benchmark: LUT, equalizeHist, cvtColor BGR<->RGB, addWeighted.
 * Compare RPP HAL (default GPU) vs native (OPENCV_RPP_DISABLE=1).
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
    Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) lut.at<uchar>(i) = (uchar)(255 - i);
    for (auto sz : { Size(1920, 1080), Size(3840, 2160) }) {
        Mat c(sz, CV_8UC3), g(sz, CV_8UC1), d;
        Mat a32(sz, CV_32FC1), b32(sz, CV_32FC1);
        randu(c, 0, 255); randu(g, 0, 255); randu(a32, 0.f, 1.f); randu(b32, 0.f, 1.f);
        cout << "--- " << sz.width << "x" << sz.height << " ---\n";
        cout << "lut3c      " << bench([&]{ LUT(c, lut, d); }, 200) << " ms\n";
        cout << "equalize1c " << bench([&]{ equalizeHist(g, d); }, 200) << " ms\n";
        cout << "bgr2rgb    " << bench([&]{ cvtColor(c, d, COLOR_BGR2RGB); }, 200) << " ms\n";
        cout << "addWt32f   " << bench([&]{ addWeighted(a32, 0.3, b32, 0.7, 0.0, d); }, 200) << " ms\n";
    }
    return 0;
}
