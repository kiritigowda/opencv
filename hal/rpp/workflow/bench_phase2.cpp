/*
 * Phase 2 op benchmark: erode, dilate, inRange, remap.
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
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
        Mat mapx(sz, CV_32FC1), mapy(sz, CV_32FC1);
        for (int y = 0; y < sz.height; ++y)
            for (int x = 0; x < sz.width; ++x) {
                mapx.at<float>(y,x) = (float)min(sz.width-1, x+2);
                mapy.at<float>(y,x) = (float)min(sz.height-1, y+1);
            }
        cout << "--- " << sz.width << "x" << sz.height << " ---\n";
        cout << "erode3x3   " << bench([&]{ erode(c, d, kernel, Point(-1,-1), 1, BORDER_REPLICATE); }, 200) << " ms\n";
        cout << "dilate3x3  " << bench([&]{ dilate(c, d, kernel, Point(-1,-1), 1, BORDER_REPLICATE); }, 200) << " ms\n";
        cout << "inRange1c  " << bench([&]{ inRange(g, Scalar(50), Scalar(200), d); }, 200) << " ms\n";
        cout << "remap3c    " << bench([&]{ remap(c, d, mapx, mapy, INTER_LINEAR, BORDER_REPLICATE); }, 100) << " ms\n";
    }
    return 0;
}
