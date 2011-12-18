#define VSTRIDE     8
#define PI  3.14159265f

void twiddle(int k, float angle, float* in, float* in_c){

     float tw, tw_c, v, v_c;
     tw = native_cos(k*angle); //Less accurate
     tw_c = native_sin(k*angle); //Less accurate
     v = tw * *in - tw_c * *in_c;
     v_c = tw * *in_c + tw_c * *in;
     *in = v; 
     *in_c = v_c;
}

void FFT_2(float* in0, float* in0_c, float* in1, float* in1_c) {

    float v, v_c;
    v = *in0;
    v_c = *in0_c;
    *in0 = v + *in1;
    *in0_c = v_c + *in1_c;
    *in1 = v - *in1;
    *in1_c = v_c - *in1_c;
}


void FFT_8(float* in0, float* in1, float* in2, float* in3,
           float* in4, float* in5, float* in6, float* in7,
           float* in_c0, float* in_c1, float* in_c2, float* in_c3,
           float* in_c4, float* in_c5, float* in_c6, float* in_c7){

    //Stage 1
    float  v0, v1, v2, v3, v4, v5, v6, v7;
    float  v_c0, v_c1, v_c2, v_c3, v_c4, v_c5, v_c6, v_c7;

    v0 = *in0 + *in4;
    v4 = *in0 - *in4;
    v2 = *in2 + *in6;
    v6 = *in_c2 - *in_c6; 
    v_c0 = *in_c0 + *in_c4;
    v_c4 = *in_c0 - *in_c4;
    v_c2 = *in_c2 + *in_c6;
    v_c6 = *in6 - *in2;

    v1 = *in1 + *in5;
    v5 = *in1 - *in5;
    v3 = *in3 + *in7;
    v7 = *in_c3 - *in_c7;
    v_c1 = *in_c1 + *in_c5;
    v_c5 = *in_c1 - *in_c5;
    v_c3 = *in_c3 + *in_c7;
    v_c7 = *in7 - *in3; 

    //Stage 2
    float w0, w1, w2, w3, w4, w5, w6, w7;
    float w_c0, w_c1, w_c2, w_c3, w_c4, w_c5, w_c6, w_c7;

    w0 = v0 + v2;
    w4 = v4 + v6;
    w2 = v0 - v2;
    w6 = v4 - v6;
    w_c0 = v_c0 + v_c2;
    w_c4 = v_c4 + v_c6;
    w_c2 = v_c0 - v_c2;
    w_c6 = v_c4 - v_c6;

    w1 = v1 + v3;
    w5 = v5 + v7;
    w3 = v1 - v3;
    w7 = v5 - v7;
    w_c1 = v_c1 + v_c3;
    w_c5 = v_c5 + v_c7;
    w_c3 = v_c1 - v_c3;
    w_c7 = v_c5 - v_c7;

    float angle = -2*PI/8;
    v0 = native_cos(1*angle);
    v_c0 = native_sin(1*angle);
    v1 = v0*w5 - v_c0*w_c5;
    v_c1 = v0*w_c5 + v_c0*w5;
    w5 = v1;
    w_c5 = v_c1;

    v2 = native_cos(2*angle);
    v_c2 = native_sin(2*angle);
    v3 = v2*w3 - v_c2*w_c3;
    v_c3 = v2*w_c3 + v_c2*w3;
    w3 = v3;
    w_c3 = v_c3;

    v4 = native_cos(3*angle);
    v_c4 = native_sin(3*angle);
    v5 = v4*w7 - v_c4*w_c7;
    v_c5 = v4*w_c7 + v_c4*w7;
    w7 = v5;
    w_c7 = v_c5;

    //Stage 3
    *in0 = w0 + w1;
    *in1 = w4 + w5;
    *in2 = w2 + w3;
    *in3 = w6 + w7;
    *in_c0 = w_c0 + w_c1;
    *in_c1 = w_c4 + w_c5;
    *in_c2 = w_c2 + w_c3;
    *in_c3 = w_c6 + w_c7;

    *in4 = w0 - w1;
    *in5 = w4 - w5;
    *in6 = w2 - w3;
    *in7 = w6 - w7;
    *in_c4 = w_c0 - w_c1;
    *in_c5 = w_c4 - w_c5;
    *in_c6 = w_c2 - w_c3;
    *in_c7 = w_c6 - w_c7;
}

#define N 1024

__kernel __attribute__((reqd_work_group_size (128,1,1))) void 
FFT_X( __global float *real_in, __global float *comp_in, __global float *dout)	{

    __local float real_lm[1098];
    __local float comp_lm[1098];
    
    //Stage 1 128 x FFT_8 
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group = (lid/VSTRIDE);

    float  in0, in1, in2, in3;
    float  in4, in5, in6, in7;
    in0 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+0*128];
    in1 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+1*128];
    in2 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+2*128];
    in3 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+3*128];
    in4 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+4*128];
    in5 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+5*128];
    in6 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+6*128];
    in7 = real_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+7*128];

    float  in_c0, in_c1, in_c2, in_c3;
    float  in_c4, in_c5, in_c6, in_c7;
    in_c0 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+0*128];
    in_c1 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+1*128];
    in_c2 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+2*128];
    in_c3 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+3*128];
    in_c4 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+4*128];
    in_c5 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+5*128];
    in_c6 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+6*128];
    in_c7 = comp_in[(lid%8)*16 + 2*((lid%64)/8) + lid/64+7*128];

    FFT_8(&in0,&in1,&in2,&in3,&in4,&in5,&in6,&in7,
          &in_c0,&in_c1,&in_c2,&in_c3,&in_c4,&in_c5,&in_c6,&in_c7);

    //Stage 2
//
    real_lm[lid%8+0*8+group*64] = in0;
    real_lm[lid%8+1*8+group*64] = in1;
    real_lm[lid%8+2*8+group*64] = in2;
    real_lm[lid%8+3*8+group*64] = in3;
    real_lm[lid%8+4*8+group*64] = in4;
    real_lm[lid%8+5*8+group*64] = in5;
    real_lm[lid%8+6*8+group*64] = in6;
    real_lm[lid%8+7*8+group*64] = in7;

    comp_lm[lid%8+0*8+group*64] = in_c0;
    comp_lm[lid%8+1*8+group*64] = in_c1;
    comp_lm[lid%8+2*8+group*64] = in_c2;
    comp_lm[lid%8+3*8+group*64] = in_c3;
    comp_lm[lid%8+4*8+group*64] = in_c4;
    comp_lm[lid%8+5*8+group*64] = in_c5;
    comp_lm[lid%8+6*8+group*64] = in_c6;
    comp_lm[lid%8+7*8+group*64] = in_c7;

    in0 = real_lm[0+lid*8];
    in1 = real_lm[1+lid*8];
    in2 = real_lm[2+lid*8];
    in3 = real_lm[3+lid*8];
    in4 = real_lm[4+lid*8];
    in5 = real_lm[5+lid*8];
    in6 = real_lm[6+lid*8];
    in7 = real_lm[7+lid*8];
    in_c0 = comp_lm[0+lid*8];
    in_c1 = comp_lm[1+lid*8];
    in_c2 = comp_lm[2+lid*8];
    in_c3 = comp_lm[3+lid*8];
    in_c4 = comp_lm[4+lid*8];
    in_c5 = comp_lm[5+lid*8];
    in_c6 = comp_lm[6+lid*8];
    in_c7 = comp_lm[7+lid*8];

    float angle = -2*PI*(lid%8)/64;
    twiddle(1, angle, &in1, &in_c1);
    twiddle(2, angle, &in2, &in_c2);
    twiddle(3, angle, &in3, &in_c3);
    twiddle(4, angle, &in4, &in_c4);
    twiddle(5, angle, &in5, &in_c5);
    twiddle(6, angle, &in6, &in_c6);
    twiddle(7, angle, &in7, &in_c7);

    //Memory Barrier
    //barrier(CLK_GLOBAL_MEM_FENCE);

    FFT_8(&in0,&in1,&in2,&in3,&in4,&in5,&in6,&in7,
          &in_c0,&in_c1,&in_c2,&in_c3,&in_c4,&in_c5,&in_c6,&in_c7);
//
//    //Stage 3
    real_lm[lid%64+0*64+(lid/64)*512] = in0;
    real_lm[lid%64+1*64+(lid/64)*512] = in1;
    real_lm[lid%64+2*64+(lid/64)*512] = in2;
    real_lm[lid%64+3*64+(lid/64)*512] = in3;
    real_lm[lid%64+4*64+(lid/64)*512] = in4;
    real_lm[lid%64+5*64+(lid/64)*512] = in5;
    real_lm[lid%64+6*64+(lid/64)*512] = in6;
    real_lm[lid%64+7*64+(lid/64)*512] = in7;

    comp_lm[lid%64+0*64+(lid/64)*512] = in_c0;
    comp_lm[lid%64+1*64+(lid/64)*512] = in_c1;
    comp_lm[lid%64+2*64+(lid/64)*512] = in_c2;
    comp_lm[lid%64+3*64+(lid/64)*512] = in_c3;
    comp_lm[lid%64+4*64+(lid/64)*512] = in_c4;
    comp_lm[lid%64+5*64+(lid/64)*512] = in_c5;
    comp_lm[lid%64+6*64+(lid/64)*512] = in_c6;
    comp_lm[lid%64+7*64+(lid/64)*512] = in_c7;

    in0 = real_lm[lid%8+0*8+group*64];
    in1 = real_lm[lid%8+1*8+group*64];
    in2 = real_lm[lid%8+2*8+group*64];
    in3 = real_lm[lid%8+3*8+group*64];
    in4 = real_lm[lid%8+4*8+group*64];
    in5 = real_lm[lid%8+5*8+group*64];
    in6 = real_lm[lid%8+6*8+group*64];
    in7 = real_lm[lid%8+7*8+group*64];
                                                
    in_c0 = comp_lm[lid%8+0*8+group*64];
    in_c1 = comp_lm[lid%8+1*8+group*64];
    in_c2 = comp_lm[lid%8+2*8+group*64];
    in_c3 = comp_lm[lid%8+3*8+group*64];
    in_c4 = comp_lm[lid%8+4*8+group*64];
    in_c5 = comp_lm[lid%8+5*8+group*64];
    in_c6 = comp_lm[lid%8+6*8+group*64];
    in_c7 = comp_lm[lid%8+7*8+group*64];

    angle = -2*PI*(lid%64)/512;
    twiddle(1, angle, &in1, &in_c1);
    twiddle(2, angle, &in2, &in_c2);
    twiddle(3, angle, &in3, &in_c3);
    twiddle(4, angle, &in4, &in_c4);
    twiddle(5, angle, &in5, &in_c5);
    twiddle(6, angle, &in6, &in_c6);
    twiddle(7, angle, &in7, &in_c7);

    //Memory Barrier
    //barrier(CLK_GLOBAL_MEM_FENCE);

    FFT_8(&in0,&in1,&in2,&in3,&in4,&in5,&in6,&in7,
          &in_c0,&in_c1,&in_c2,&in_c3,&in_c4,&in_c5,&in_c6,&in_c7);


    real_lm[lid%64+0*64+(lid/64)*512] = in0;
    real_lm[lid%64+1*64+(lid/64)*512] = in1;
    real_lm[lid%64+2*64+(lid/64)*512] = in2;
    real_lm[lid%64+3*64+(lid/64)*512] = in3;
    real_lm[lid%64+4*64+(lid/64)*512] = in4;
    real_lm[lid%64+5*64+(lid/64)*512] = in5;
    real_lm[lid%64+6*64+(lid/64)*512] = in6;
    real_lm[lid%64+7*64+(lid/64)*512] = in7;

    comp_lm[lid%64+0*64+(lid/64)*512] = in_c0;
    comp_lm[lid%64+1*64+(lid/64)*512] = in_c1;
    comp_lm[lid%64+2*64+(lid/64)*512] = in_c2;
    comp_lm[lid%64+3*64+(lid/64)*512] = in_c3;
    comp_lm[lid%64+4*64+(lid/64)*512] = in_c4;
    comp_lm[lid%64+5*64+(lid/64)*512] = in_c5;
    comp_lm[lid%64+6*64+(lid/64)*512] = in_c6;
    comp_lm[lid%64+7*64+(lid/64)*512] = in_c7;

    in0 = real_lm[lid+0*128];
    in1 = real_lm[lid+1*128];
    in2 = real_lm[lid+2*128];
    in3 = real_lm[lid+3*128];
    in4 = real_lm[lid+4*128];
    in5 = real_lm[lid+5*128];
    in6 = real_lm[lid+6*128];
    in7 = real_lm[lid+7*128];
                                                
    in_c0 = comp_lm[lid+0*128];
    in_c1 = comp_lm[lid+1*128];
    in_c2 = comp_lm[lid+2*128];
    in_c3 = comp_lm[lid+3*128];
    in_c4 = comp_lm[lid+4*128];
    in_c5 = comp_lm[lid+5*128];
    in_c6 = comp_lm[lid+6*128];
    in_c7 = comp_lm[lid+7*128];

    angle = -2*PI*(lid+0*128)/1024;
    twiddle(1, angle, &in4, &in_c4);

    angle = -2*PI*(lid+1*128)/1024;
    twiddle(1, angle, &in5, &in_c5);

    angle = -2*PI*(lid+2*128)/1024;
    twiddle(1, angle, &in6, &in_c6);

    angle = -2*PI*(lid+3*128)/1024;
    twiddle(1, angle, &in7, &in_c7);
    
    FFT_2(&in0, &in_c0, &in4, &in_c4);
    FFT_2(&in1, &in_c1, &in5, &in_c5);
    FFT_2(&in2, &in_c2, &in6, &in_c6);
    FFT_2(&in3, &in_c3, &in7, &in_c7);
    
    dout[lid+0*128] = in0;
    dout[lid+1*128] = in1;
    dout[lid+2*128] = in2;
    dout[lid+3*128] = in3;
    dout[lid+4*128] = in4;
    dout[lid+5*128] = in5;
    dout[lid+6*128] = in6;
    dout[lid+7*128] = in7;

    dout[lid+8*128] = in_c0;
    dout[lid+9*128] = in_c1;
    dout[lid+10*128] = in_c2;
    dout[lid+11*128] = in_c3;
    dout[lid+12*128] = in_c4;
    dout[lid+13*128] = in_c5;
    dout[lid+14*128] = in_c6;
    dout[lid+15*128] = in_c7;
}

__kernel __attribute__((reqd_work_group_size (128,1,1))) void
FFT_TWO( __global float *real_in, __global float *comp_in, __global float *dout)	{

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    float in0, in1, in_c0, in_c1;
    float in2, in3, in_c2, in_c3;
    float in4, in5, in_c4, in_c5;
    float in6, in7, in_c6, in_c7;

    in0 = real_in[lid];
    in1 = real_in[lid+1*128];
    in2 = real_in[lid+2*128];
    in3 = real_in[lid+3*128];
    in4 = real_in[lid+4*128];
    in5 = real_in[lid+5*128];
    in6 = real_in[lid+6*128];
    in7 = real_in[lid+7*128];

    in_c0 = comp_in[lid];
    in_c1 = comp_in[lid+1*128];
    in_c2 = comp_in[lid+2*128];
    in_c3 = comp_in[lid+3*128];
    in_c4 = comp_in[lid+4*128];
    in_c5 = comp_in[lid+5*128];
    in_c6 = comp_in[lid+6*128];
    in_c7 = comp_in[lid+7*128];


    FFT_2(&in0, &in_c0, &in4, &in_c4);
    FFT_2(&in1, &in_c1, &in5, &in_c5);
    FFT_2(&in2, &in_c2, &in6, &in_c6);
    FFT_2(&in3, &in_c3, &in7, &in_c7);

    dout[lid+0*128] = in0;
    dout[lid+1*128] = in1;
    dout[lid+2*128] = in2;
    dout[lid+3*128] = in3;
    dout[lid+4*128] = in4;
    dout[lid+5*128] = in5;
    dout[lid+6*128] = in6;
    dout[lid+7*128] = in7;

    dout[lid+8*128] = in_c0;
    dout[lid+9*128] = in_c1;
    dout[lid+10*128] = in_c2;
    dout[lid+11*128] = in_c3;
    dout[lid+12*128] = in_c4;
    dout[lid+13*128] = in_c5;
    dout[lid+14*128] = in_c6;
    dout[lid+15*128] = in_c7;

}

__kernel __attribute__((reqd_work_group_size (128,1,1))) void
FFT_TWIDDLE( __global float *real_in, __global float *comp_in, __global float *dout) {


    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    float in0, in1, in_c0, in_c1;
    float in2, in3, in_c2, in_c3;
    float in4, in5, in_c4, in_c5;
    float in6, in7, in_c6, in_c7;

    in0 = real_in[lid];
    in1 = real_in[lid+1*128];
    in2 = real_in[lid+2*128];
    in3 = real_in[lid+3*128];
    in4 = real_in[lid+4*128];
    in5 = real_in[lid+5*128];
    in6 = real_in[lid+6*128];
    in7 = real_in[lid+7*128];

    in_c0 = comp_in[lid];
    in_c1 = comp_in[lid+1*128];
    in_c2 = comp_in[lid+2*128];
    in_c3 = comp_in[lid+3*128];
    in_c4 = comp_in[lid+4*128];
    in_c5 = comp_in[lid+5*128];
    in_c6 = comp_in[lid+6*128];
    in_c7 = comp_in[lid+7*128];

    float angle = -2*PI*(lid+0*128)/2048;
    twiddle(1, angle, &in0, &in_c0);

    angle = -2*PI*(lid+1*128)/2048;
    twiddle(1, angle, &in1, &in_c1);

    angle = -2*PI*(lid+2*128)/2048;
    twiddle(1, angle, &in2, &in_c2);

    angle = -2*PI*(lid+3*128)/2048;
    twiddle(1, angle, &in3, &in_c3);

    angle = -2*PI*(lid+4*128)/2048;
    twiddle(1, angle, &in4, &in_c4);

    angle = -2*PI*(lid+5*128)/2048;
    twiddle(1, angle, &in5, &in_c5);

    angle = -2*PI*(lid+6*128)/2048;
    twiddle(1, angle, &in6, &in_c6);

    angle = -2*PI*(lid+7*128)/2048;
    twiddle(1, angle, &in7, &in_c7);

    dout[lid+0*128] = in0;
    dout[lid+1*128] = in1;
    dout[lid+2*128] = in2;
    dout[lid+3*128] = in3;
    dout[lid+4*128] = in4;
    dout[lid+5*128] = in5;
    dout[lid+6*128] = in6;
    dout[lid+7*128] = in7;

    dout[lid+8*128] = in_c0;
    dout[lid+9*128] = in_c1;
    dout[lid+10*128] = in_c2;
    dout[lid+11*128] = in_c3;
    dout[lid+12*128] = in_c4;
    dout[lid+13*128] = in_c5;
    dout[lid+14*128] = in_c6;
    dout[lid+15*128] = in_c7;
}
__kernel __attribute__((reqd_work_group_size (2048,1,1))) void
CONJM( __global float *inA, __global float *inB, __global float *dout) {

    uint lid = get_local_id(0);

    float in0, in_c0, in1, in_c1;
    in0 = inA[lid];
    in_c0 = inA[lid+2048];
    in1 = inB[lid];
    in_c1 = inB[lid+2048];

    float in_c1_cc;
    in_c1_cc = -1*in_c1;

    float out, out_c;
    out = in0*in1-in_c0*in_c1_cc;
    out_c = in0*in_c1_cc + in1*in_c0;     

    dout[lid] = out;
    dout[lid+2048] = -1*out_c; //Additional Conj for ifft
}

__kernel __attribute__((reqd_work_group_size (1024,1,1))) void
IFFT_SCALE( __global float *in,  __global float *dout) {

    uint lid = get_local_id(0);

    float in0, in1, in_c0, in_c1;
    in0 = in[lid];
    in1 = in[lid+1024];
    in_c0 = in[lid+2048];
    in_c1 = in[lid+3072];

    float out0, out1, out_c0, out_c1;
    out0 = in0/4194304;
    out1 = in1/4194304;
    out_c0 = in_c0/4194304; 
    out_c1 = in_c1/4194304; 
    
    //Shift output to center 0 freq
    dout[lid] = pow(out1,2);
    dout[lid+1024] = pow(out0,2);
    dout[lid+2048] = -1*out_c1;
    dout[lid+3072] = -1*out_c0;
    
}

__kernel __attribute__((reqd_work_group_size (2048,1,1))) void
ILQL_GEN( __global float *inPRN, __global float *inPRNTime, float freq, __global float *dout) {

    uint lid = get_local_id(0);

    float in, in_time;
    in = inPRN[lid];
    in_time = inPRNTime[lid];

    float bpsk, il_out, ql_out;
    bpsk = -1*(in*2-1);
    il_out = sqrt((float) 2)*native_sin(2*PI*(freq)*in_time)*bpsk;
    ql_out = sqrt((float) 2)*native_cos(2*PI*(freq)*in_time)*bpsk;

    dout[lid] = il_out;
    dout[lid+2048] = ql_out;
}
