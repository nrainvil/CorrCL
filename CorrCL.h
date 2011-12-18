#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <SDKFile.hpp>
#include <SDKCommon.hpp>
#include <SDKApplication.hpp>
#include <utility>
#include <string>
#include <iterator>
#include <CL/cl.hpp>
#include <cmath>
#include <sys/time.h>

#define __NO_STD_STRING 
#define __CL_ENABLE_EXCEPTIONS
#define PI 3.14159265f
using namespace cl;

class CorrCL {
    public:
    CorrCL(int VSIZE);
    float* fft2048(float* in, float* in_i);
    float** fft_m2048(float** in_m, float** in_i_m, int len);
    float* ifft2048(float* inAinit);
    float** ifft_m2048(float** inA_m, int len);
    float* cm2048(float* inA, float* inB);
    float** cm_m2048(float** inA_m, float* inB, int len);
    float* ilql2048(float* inPrn, float* inPrnTime, float freq);
    float** ilql_m2048(float* inPrn, float* inPrnTime, float* d_v, float freq, int len);

    private:
    int VSIZE;
    cl::CommandQueue queue;
    cl::Context context;
    cl::Kernel kernel1;
    cl::Kernel kernel2;
    cl::Kernel kernelTwiddle;
    cl::Kernel kernel3;
    cl::Kernel kernel4;
    cl::Kernel kernelcm;
    cl::Kernel kernelilql;
    cl::Kernel kernelis;
};
