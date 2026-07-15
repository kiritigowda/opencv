/*
 * Comprehensive OpenCV-API benchmark over every operation the RPP HAL
 * integrates (23 ops). Because the HAL intercepts cv:: calls transparently,
 * running this binary under different env toggles measures each backend:
 *
 *   OPENCV_RPP_DISABLE=1   -> OpenCV native (AVX)          [config: native]
 *   OPENCV_RPP_FORCE_CPU=1 -> RPP HOST via the HAL         [config: hal-cpu]
 *   (default)              -> RPP HIP via the HAL          [config: hal-hip]
 *
 * Fixed size 1920x1080. Prints one row per op: name, ms/op.
 * Args: [iters] [warmups]  (defaults 100 / 5).
 */
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

static int ITERS = 100, WARMUPS = 5;

static double ms(chrono::steady_clock::time_point a, chrono::steady_clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() / 1000.0;
}

struct Case { string name; function<void()> fn; };

int main(int argc, char** argv) {
    if (argc > 1) ITERS = atoi(argv[1]);
    if (argc > 2) WARMUPS = atoi(argv[2]);

    const int W = 1920, H = 1080;
    RNG rng(42);
    Mat g(H, W, CV_8UC1), g2(H, W, CV_8UC1), c(H, W, CV_8UC3);
    Mat xf(H, W, CV_32FC1), yf(H, W, CV_32FC1), af(H, W, CV_32FC1), bf(H, W, CV_32FC1);
    rng.fill(g, RNG::UNIFORM, 0, 255);   rng.fill(g2, RNG::UNIFORM, 0, 255);
    rng.fill(c, RNG::UNIFORM, 0, 255);
    rng.fill(xf, RNG::UNIFORM, -10.f, 10.f); rng.fill(yf, RNG::UNIFORM, -10.f, 10.f);
    rng.fill(af, RNG::UNIFORM, 0.f, 1.f);    rng.fill(bf, RNG::UNIFORM, 0.f, 1.f);

    double Ma[6] = {1, 0.05, 30, 0.02, 1, 20};      Mat Maff(2, 3, CV_64F, Ma);
    double Mp[9] = {1, 0.05, 10, 0.02, 1, 20, 1e-4, 2e-4, 1}; Mat Mper(3, 3, CV_64F, Mp);
    Mat lut(1, 256, CV_8UC1); for (int i = 0; i < 256; ++i) lut.at<uchar>(i) = (uchar)(255 - i);
    Mat kern = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat mapx(H, W, CV_32FC1), mapy(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        mapx.at<float>(y,x) = (float)min(W-1, x+2); mapy.at<float>(y,x) = (float)min(H-1, y+1);
    }

    Mat d; Scalar s, m, sd;
    vector<Case> cases = {
        {"bitwise_and 8UC1",  [&]{ bitwise_and(g, g2, d); }},
        {"bitwise_or 8UC1",   [&]{ bitwise_or(g, g2, d); }},
        {"bitwise_xor 8UC1",  [&]{ bitwise_xor(g, g2, d); }},
        {"bitwise_not 8UC1",  [&]{ bitwise_not(g, d); }},
        {"boxFilter 3x3 8UC3",[&]{ boxFilter(c, d, -1, Size(3,3), Point(-1,-1), true, BORDER_REPLICATE); }},
        {"gaussianBlur 3x3",  [&]{ GaussianBlur(c, d, Size(3,3), 0, 0, BORDER_REPLICATE); }},
        {"medianBlur 3x3",    [&]{ medianBlur(c, d, 3); }},
        {"sobel dx 3x3 8UC1", [&]{ Sobel(g, d, CV_8U, 1, 0, 3, 1, 0, BORDER_REPLICATE); }},
        {"resize down2x 8UC3",[&]{ resize(c, d, Size(W/2, H/2), 0, 0, INTER_LINEAR); }},
        {"warpAffine 8UC3",   [&]{ warpAffine(c, d, Maff, c.size(), INTER_LINEAR, BORDER_REPLICATE); }},
        {"warpPerspect 8UC3", [&]{ warpPerspective(c, d, Mper, c.size(), INTER_LINEAR, BORDER_REPLICATE); }},
        {"flip horiz 8UC3",   [&]{ flip(c, d, 1); }},
        {"erode 3x3 8UC3",    [&]{ erode(c, d, kern, Point(-1,-1), 1, BORDER_REPLICATE); }},
        {"dilate 3x3 8UC3",   [&]{ dilate(c, d, kern, Point(-1,-1), 1, BORDER_REPLICATE); }},
        {"remap 8UC3 bilin",  [&]{ remap(c, d, mapx, mapy, INTER_LINEAR, BORDER_REPLICATE); }},
        {"inRange 8UC1",      [&]{ inRange(g, Scalar(50), Scalar(200), d); }},
        {"lut 8UC3",          [&]{ LUT(c, lut, d); }},
        {"equalizeHist 8UC1", [&]{ equalizeHist(g, d); }},
        {"cvtColor BGR2RGB",  [&]{ cvtColor(c, d, COLOR_BGR2RGB); }},
        {"addWeighted 32f",   [&]{ addWeighted(af, 0.3, bf, 0.7, 0.0, d); }},
        {"sum 8UC1",          [&]{ s = sum(g); }},
        {"meanStdDev 8UC1",   [&]{ meanStdDev(g, m, sd); }},
        {"magnitude 32f",     [&]{ magnitude(xf, yf, d); }},
    };

    cout << left << setw(22) << "op" << right << setw(12) << "ms/op" << "\n";
    cout << string(34, '-') << "\n";
    for (auto& cs : cases) {
        try {
            for (int i = 0; i < WARMUPS; ++i) cs.fn();
            auto t1 = chrono::steady_clock::now();
            for (int i = 0; i < ITERS; ++i) cs.fn();
            auto t2 = chrono::steady_clock::now();
            cout << left << setw(22) << cs.name << right << fixed << setprecision(4)
                 << setw(12) << (ms(t1, t2) / ITERS) << "\n";
        } catch (const std::exception& e) {
            cout << left << setw(22) << cs.name << right << setw(12) << "ERR" << "\n";
        }
    }
    return 0;
}
