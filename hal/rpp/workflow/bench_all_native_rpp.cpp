/*
 * Native RPP benchmark (no OpenCV integration) over the same 23 kernels the
 * RPP HAL uses. Measures pure kernel time: memory is uploaded once before the
 * timed loop and downloaded once after, so this reflects RPP's raw compute
 * potential (the HAL additionally pays a host<->device copy per call).
 *
 *   ./bench_all_native_rpp            -> RPP HOST backend   [config: rpp-cpu]
 *   OPENCV_RPP_FORCE_GPU=1 ./...      -> RPP HIP  backend   [config: rpp-hip]
 *
 * Fixed size 1920x1080. Args: [iters] [warmups] (defaults 100 / 5).
 */
#include <rpp/rppt.h>
#include <rpp/rppt_tensor_bitwise_operations.h>
#include <rpp/rppt_tensor_filter_augmentations.h>
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <rpp/rppt_tensor_morphological_operations.h>
#include <rpp/rppt_tensor_statistical_operations.h>
#include <rpp/rppt_tensor_color_augmentations.h>
#include <rpp/rppt_tensor_data_exchange_operations.h>
#include <rpp/rppt_tensor_arithmetic_operations.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <functional>
#include <cstring>
#include <cstdlib>
using namespace std;

static int ITERS = 100, WARMUPS = 5;
static bool USE_HIP = false;

static double ms(chrono::steady_clock::time_point a, chrono::steady_clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() / 1000.0;
}

// A backend-agnostic buffer: device memory for HIP, host malloc for HOST.
static void* devAlloc(size_t n) {
    void* p = nullptr;
    if (USE_HIP) { if (hipMalloc(&p, n) != hipSuccess) return nullptr; }
    else p = malloc(n);
    return p;
}
static void devFree(void* p) { if (!p) return; if (USE_HIP) (void)hipFree(p); else free(p); }

static void descNHWC(RpptDesc& d, int w, int h, int c, RpptDataType dt) {
    memset(&d, 0, sizeof(d));
    d.numDims = 4; d.dataType = dt; d.n = 1; d.c = c; d.h = h; d.w = w; d.layout = NHWC;
    d.strides.wStride = c; d.strides.hStride = (size_t)w * c; d.strides.cStride = 1;
    d.strides.nStride = (size_t)w * h * c;
}
static void fullRoi(RpptROI& r, int w, int h) {
    r.xywhROI.xy.x = 0; r.xywhROI.xy.y = 0; r.xywhROI.roiWidth = w; r.xywhROI.roiHeight = h;
}

struct Case { string name; function<RppStatus(rppHandle_t, RppBackend)> fn; };

int main(int argc, char** argv) {
    if (argc > 1) ITERS = atoi(argv[1]);
    if (argc > 2) WARMUPS = atoi(argv[2]);
    const char* f = getenv("OPENCV_RPP_FORCE_GPU");
    USE_HIP = f && (!strcmp(f,"1") || !strcmp(f,"yes") || !strcmp(f,"true"));
    RppBackend be = USE_HIP ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;

    if (USE_HIP) { int n=0; if (hipGetDeviceCount(&n)!=hipSuccess || n<=0) { cerr<<"no HIP device\n"; return 1; } }

    const int W = 1920, H = 1080;
    const size_t n1 = (size_t)W*H, n3 = n1*3;

    // Buffers (U8 1ch, U8 3ch, F32 1ch x/y/dst, resize dst).
    void *g=devAlloc(n1), *g2=devAlloc(n1), *d1=devAlloc(n1);
    void *c=devAlloc(n3), *d3=devAlloc(n3), *d3b=devAlloc(n3);
    void *xf=devAlloc(n1*4), *yf=devAlloc(n1*4), *df=devAlloc(n1*4);
    void *dResize=devAlloc(n3);           // down2x fits in n3
    void *lutHost=malloc(65536); memset(lutHost,0,65536);
    for (int i=0;i<256;++i) ((unsigned char*)lutHost)[i]=(unsigned char)(255-i);
    // small param/result buffers
    void *affine=devAlloc(6*4), *persp=devAlloc(9*4), *perm=devAlloc(3*4);
    void *mapRow=devAlloc(n1*4), *mapCol=devAlloc(n1*4);
    void *redRes=devAlloc(64), *meanRes=devAlloc(64);
    { float aff[6]={1,0.05f,30,0.02f,1,20}; float pp[9]={1,0.05f,10,0.02f,1,20,1e-4f,2e-4f,1}; unsigned int pm[3]={2,1,0};
      if (USE_HIP){ hipMemcpy(affine,aff,24,hipMemcpyHostToDevice); hipMemcpy(persp,pp,36,hipMemcpyHostToDevice); hipMemcpy(perm,pm,12,hipMemcpyHostToDevice);}
      else { memcpy(affine,aff,24); memcpy(persp,pp,36); memcpy(perm,pm,12);} }

    RpptDesc u1, u3, f1, u3half; descNHWC(u1,W,H,1,U8); descNHWC(u3,W,H,3,U8);
    descNHWC(f1,W,H,1,F32); descNHWC(u3half,W/2,H/2,3,U8);
    RpptROI roi, roiHalf; fullRoi(roi,W,H); fullRoi(roiHalf,W/2,H/2);
    RpptImagePatch patchHalf; patchHalf.width=W/2; patchHalf.height=H/2;
    Rpp32u hz=1, vt=0; float stddev=1.0f; float alpha=0.3f;

    vector<Case> cases = {
      {"bitwise_and 8UC1",[&](rppHandle_t h,RppBackend b){ return rppt_bitwise_and(g,g2,&u1,d1,&u1,&roi,XYWH,h,b);} },
      {"bitwise_or 8UC1", [&](rppHandle_t h,RppBackend b){ return rppt_bitwise_or(g,g2,&u1,d1,&u1,&roi,XYWH,h,b);} },
      {"bitwise_xor 8UC1",[&](rppHandle_t h,RppBackend b){ return rppt_bitwise_xor(g,g2,&u1,d1,&u1,&roi,XYWH,h,b);} },
      {"bitwise_not 8UC1",[&](rppHandle_t h,RppBackend b){ return rppt_bitwise_not(g,&u1,d1,&u1,&roi,XYWH,h,b);} },
      {"boxFilter 3x3 8UC3",[&](rppHandle_t h,RppBackend b){ return rppt_box_filter(c,&u3,d3,&u3,3,REPLICATE,&roi,XYWH,h,b);} },
      {"gaussianBlur 3x3",[&](rppHandle_t h,RppBackend b){ return rppt_gaussian_filter(c,&u3,d3,&u3,&stddev,3,REPLICATE,&roi,XYWH,h,b);} },
      {"medianBlur 3x3",  [&](rppHandle_t h,RppBackend b){ return rppt_median_filter(c,&u3,d3,&u3,3,REPLICATE,&roi,XYWH,h,b);} },
      {"sobel dx 3x3 8UC1",[&](rppHandle_t h,RppBackend b){ return rppt_sobel_filter(g,&u1,d1,&u1,0,3,&roi,XYWH,h,b);} },
      {"resize down2x 8UC3",[&](rppHandle_t h,RppBackend b){ return rppt_resize(c,&u3,dResize,&u3half,&patchHalf,BILINEAR,&roi,XYWH,h,b);} },
      {"warpAffine 8UC3", [&](rppHandle_t h,RppBackend b){ return rppt_warp_affine(c,&u3,d3,&u3,(Rpp32f*)affine,BILINEAR,&roi,XYWH,h,b);} },
      {"warpPerspect 8UC3",[&](rppHandle_t h,RppBackend b){ return rppt_warp_perspective(c,&u3,d3,&u3,(Rpp32f*)persp,BILINEAR,&roi,XYWH,h,b);} },
      {"flip horiz 8UC3", [&](rppHandle_t h,RppBackend b){ return rppt_flip(c,&u3,d3,&u3,&hz,&vt,&roi,XYWH,h,b);} },
      {"erode 3x3 8UC3",  [&](rppHandle_t h,RppBackend b){ return rppt_erode(c,&u3,d3,&u3,3,&roi,XYWH,h,b);} },
      {"dilate 3x3 8UC3", [&](rppHandle_t h,RppBackend b){ return rppt_dilate(c,&u3,d3,&u3,3,&roi,XYWH,h,b);} },
      {"remap 8UC3 bilin",[&](rppHandle_t h,RppBackend b){ return rppt_remap(c,&u3,d3,&u3,(Rpp32f*)mapRow,(Rpp32f*)mapCol,&f1,BILINEAR,&roi,XYWH,h,b);} },
      {"inRange 8UC1",    [&](rppHandle_t h,RppBackend b){ static float lo=50,hi=200; return rppt_threshold(g,&u1,d1,&u1,&lo,&hi,&roi,XYWH,h,b);} },
      {"lut 8UC3",        [&](rppHandle_t h,RppBackend b){ return rppt_lut(c,&u3,d3,&u3,lutHost,&roi,XYWH,h,b);} },
      {"equalizeHist 8UC1",[&](rppHandle_t h,RppBackend b){ return rppt_histogram_equalize(g,&u1,d1,&u1,&roi,XYWH,h,b);} },
      {"cvtColor BGR2RGB",[&](rppHandle_t h,RppBackend b){ return rppt_channel_permute(c,&u3,d3,&u3,(Rpp32u*)perm,h,b);} },
      {"addWeighted 32f", [&](rppHandle_t h,RppBackend b){ return rppt_blend(xf,yf,&f1,df,&f1,&alpha,&roi,XYWH,h,b);} },
      {"sum 8UC1",        [&](rppHandle_t h,RppBackend b){ return rppt_tensor_sum(g,&u1,redRes,1,&roi,XYWH,h,b);} },
      {"meanStdDev 8UC1", [&](rppHandle_t h,RppBackend b){ RppStatus s=rppt_tensor_mean(g,&u1,meanRes,1,&roi,XYWH,h,b); if(s!=RPP_SUCCESS) return s; return rppt_tensor_stddev(g,&u1,redRes,1,(Rpp32f*)meanRes,&roi,XYWH,h,b);} },
      {"magnitude 32f",   [&](rppHandle_t h,RppBackend b){ return rppt_magnitude(xf,yf,&f1,df,&f1,&roi,XYWH,h,b);} },
    };

    rppHandle_t handle=nullptr;
    if (USE_HIP) (void)hipGetLastError();   // clear pre-existing sticky error
    if (rppCreate(&handle,1,0,nullptr,be)!=rppStatusSuccess || !handle){ cerr<<"rppCreate failed\n"; return 1; }
    if (USE_HIP) (void)hipGetLastError();   // RPP HIP handle-create leaves a benign sticky error

    cout << left << setw(22) << "op" << right << setw(12) << "ms/op" << setw(10) << "status" << "\n";
    cout << string(44,'-') << "\n";
    for (auto& cs : cases) {
        bool bad=false;
        // RPP 3.x HIP leaves a sticky async error after each kernel that the
        // next launch mis-reads as a failure; clear it per call (as the HAL does).
        try {
            for(int i=0;i<WARMUPS;++i){
                RppStatus s=cs.fn(handle,be);
                if(USE_HIP){ hipDeviceSynchronize(); (void)hipGetLastError(); }
                if(s!=RPP_SUCCESS) bad=true;
            }
        } catch(...) { bad=true; }
        double per=0;
        if(!bad){
            auto t1=chrono::steady_clock::now();
            for(int i=0;i<ITERS;++i){ cs.fn(handle,be); if(USE_HIP){ hipDeviceSynchronize(); (void)hipGetLastError(); } }
            auto t2=chrono::steady_clock::now();
            per=ms(t1,t2)/ITERS;
        }
        cout << left << setw(22) << cs.name << right << fixed << setprecision(4)
             << setw(12) << (bad?0.0:per) << setw(10) << (bad?"ERR":"OK") << "\n";
        if(USE_HIP) (void)hipGetLastError();
    }
    rppDestroy(handle,be);
    return 0;
}
