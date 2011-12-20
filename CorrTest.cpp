/*
Copyright (c) 2011, Nicholas Rarinville
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
The name of its contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CorrCL.h"
#include <iostream>

int main () {
    int VSIZE = 1024;
    //Create Correlator Object
    CorrCL cc = CorrCL(VSIZE);

    //Read Signal Input File
	float *inA = new float[2*VSIZE];
	float *inA_i = new float[2*VSIZE];
    char * pEnd;

    std::ifstream file("hw2_data.dat");
    if (file.is_open()) {
        std::string s;
        int i = 0;
        while (getline(file, s) && i<2*VSIZE){
            inA[i] = strtod(s.c_str(), &pEnd);
            i++;
        }
    }

    //Generate imaginary components
	for (int i=0;i<2*VSIZE;i++){
		inA_i[i] = 0;
	}
    
    //Generate Quadrature and Inphase signals for PRN22
    float *inPrn22 = new float[2*VSIZE];
    std::ifstream file_prn("prn_22.dat");
    if (file_prn.is_open()) {
        std::string s;
        int i = 0;
        while (getline(file_prn, s) && i<2*VSIZE){
            inPrn22[i] = strtod(s.c_str(), &pEnd);
            i++;
        }
    }
    float *inPrn22Time = new float[2*VSIZE];
    std::ifstream file_prn_time("prn_22_time.dat");
    if (file_prn_time.is_open()) {
        std::string s;
        int i = 0;
        while (getline(file_prn_time, s) && i<2*VSIZE){
            inPrn22Time[i] = strtod(s.c_str(), &pEnd);
            i++;
        }
    }
    
    float freq = 1405000;
    float freq_d = 0;
    float delta_freq = 22923.66974755583;
    float dop_bin = 500;
    float bpsk;
    float *d_v = new float[92];
    float incr_freq = -1*delta_freq;
    for (int i=0;i<92;i++) {
        d_v[i] = incr_freq;
        incr_freq = incr_freq + dop_bin;
    } 

    //Init loop variables
    float inFreq = 0;
    float *Ql22 = new float[2*VSIZE];
    float *Il22 = new float[2*VSIZE];
    float *fftA = new float[4*VSIZE];
    float *fftIl = new float[4*VSIZE];
    float *fftQl = new float[4*VSIZE];
    float *IlQl = new float[4*VSIZE];
    float *cmIl_A = new float[4*VSIZE];
    float *cmQl_A = new float[4*VSIZE];
    float *outIl_sq = new float[4*VSIZE];
    float *outQl_sq = new float[4*VSIZE];

    float Z_max_sq = 0;
    float Z_mag_sq = 0;
    float max_time_shift = 0;
    float max_dop = 0;

    //Start Timer
    struct timeval s_tv;
    gettimeofday(&s_tv, NULL);

    float **IlQl_m = new float*[92];
    float **Il22_m = new float*[92];
    float **Ql22_m = new float*[92];
    float **Il22_comp_m = new float*[92];
    float **Ql22_comp_m = new float*[92];
    float **cmIl_A_m = new float*[92];
    float **cmQl_A_m = new float*[92];
    float **outIl_sq_m = new float*[92];
    float **outQl_sq_m = new float*[92];

    //Calculate Inphase and Quadrature Signals
    IlQl_m = cc.ilql_m2048(inPrn22, inPrn22Time, d_v, freq, 92);

    for(int j=0;j<92;j++){
        IlQl = IlQl_m[j];
        Il22_m[j] = &IlQl[0];
        Il22_comp_m[j] = &inA_i[0];
        Ql22_m[j] = &IlQl[2*VSIZE];
        Ql22_comp_m[j] = &inA_i[0];
    }

    //Calculate FFT of input an IL, Ql
    float **fftIl_m = new float*[92];
    float **fftQl_m = new float*[92];
    fftA = cc.fft2048(inA, inA_i);
    fftIl_m = cc.fft_m2048(Il22_m, Il22_comp_m, 92);
    fftQl_m = cc.fft_m2048(Ql22_m, Ql22_comp_m, 92);

    //Calculate Conjugate multiply - Also additional Conj
    cmIl_A_m = cc.cm_m2048(fftIl_m, fftA, 92);
    cmQl_A_m = cc.cm_m2048(fftQl_m, fftA, 92);


    outIl_sq_m = cc.ifft_m2048(cmIl_A_m, 92);
    outQl_sq_m = cc.ifft_m2048(cmQl_A_m, 92);

    //Start Correlation loop
    for(int j=0;j<92;j++){ //92
        freq_d = d_v[j];

        //Calculate Inverse FFT, Square result
//        outIl_sq = cc.ifft2048(cmIl_A_m[j]);
//        outQl_sq = cc.ifft2048(cmQl_A_m[j]);

        //Calculate magnitude of Z
        //Z = Il + Ql*i, imaginary component known to be zero for both
        //Mag(Z) = (Il^2 + Ql^2)^(1/2), Il and Ql are already squared by ifft func
        //Max is proportional to Il^2+Ql^2
        for (int h=0;h<2*VSIZE;h++) {
            outIl_sq = outIl_sq_m[j];
            outQl_sq = outQl_sq_m[j];
            Z_mag_sq = outIl_sq[h] + outQl_sq[h]; 
            if (Z_mag_sq>Z_max_sq) {
                Z_max_sq = Z_mag_sq;
                max_time_shift = h;
                max_dop = freq_d;
            }
        }
    }

    double max_time = (max_time_shift - VSIZE)/5714286;
    struct timeval e_tv;
    gettimeofday(&e_tv, NULL);
   
    //Print Results
//    printf("Input\n");
//    for(int i=0;i<2*VSIZE;i++) {
//        printf("%3d:%9f:",i,inA[i]);
//        if ((i+1)%256==0) printf("\n");
//    }
//    printf("\nImaginary\n");
//    for(int i=0;i<2*VSIZE;i++) {
//        printf("%3d:%9f:",i,inA_i[i]);
//        if ((i+1)%256==0) printf("\n");
//    }
//
//    float* outPrint = new float[4*VSIZE];
//    outPrint = outIl_sq_m[91];
//    printf("\nOutput - Freq_d:%f\n", freq_d);
//    for(int i=0;i<4*VSIZE;i++) {
//        if (i==2*VSIZE) printf("\nImaginary\n");
//        printf("%3d:%9f:",i%(2*VSIZE),outPrint[i]);
//        if ((i+1)%256==0) printf("\n");
//    }

    long int ft = e_tv.tv_sec - s_tv.tv_sec;
    double timediff = ((float) e_tv.tv_usec)/1000000 - ((float) s_tv.tv_usec)/1000000 + ((float) ft);
    printf("\nTime Start (s) %Ld.%06Ld\n", s_tv.tv_sec, s_tv.tv_usec);
    printf("Time End (s) %Ld.%06Ld\n", e_tv.tv_sec, e_tv.tv_usec);
    printf("Run Time (s) %06Lf\n", timediff);

    printf("\nMax Z: %f\nTime: %f\nDoppler Freq:%f\n", sqrt(Z_max_sq), max_time, max_dop);


}

