/*
 * Phase 5/6 op benchmark: sum, meanStdDev, magnitude.
 * Compare RPP HAL (default GPU) vs native (OPENCV_RPP_DISABLE=1).
 */
#include <opencv2/core.hpp>
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
        Mat g(sz, CV_8UC1), x(sz, CV_32FC1), y(sz, CV_32FC1), d;
        Scalar s, m, sd;
        randu(g, 0, 255); randu(x, -10.f, 10.f); randu(y, -10.f, 10.f);
        cout << "--- " << sz.width << "x" << sz.height << " ---\n";
        cout << "sum1c      " << bench([&]{ s = sum(g); }, 200) << " ms\n";
        cout << "meanStdDev " << bench([&]{ meanStdDev(g, m, sd); }, 200) << " ms\n";
        cout << "magnitude  " << bench([&]{ magnitude(x, y, d); }, 200) << " ms\n";
    }
    return 0;
}
