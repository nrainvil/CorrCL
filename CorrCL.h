/*
Copyright (c) 2011, Nicholas Rarinville
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
The name of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
