#include "CorrCL.h"

CorrCL::CorrCL(int VSIZE_in):VSIZE(VSIZE_in)  {
    const std::string kernel_filename = "CorrCL_kernels.cl";
	std::string kernel_2 = "FFT_TWO";
	std::string kernel_1024 = "FFT_X";
    std::string kernel_twiddle = "FFT_TWIDDLE";
    std::string kernel_cm = "CONJM";
    std::string kernel_ilql = "ILQL_GEN";
    std::string kernel_is = "IFFT_SCALE";

	//Input Data
	try {
		cl_int err;
		//Get Platform info
    		std::vector< cl::Platform > platforms; //Create vector containing type Platform
    		std::vector< cl::Platform >::iterator platform; //Create iterator 
    		cl::Platform::get(&platforms);
    		if(platforms.size() < 1) {
        		printf("ERR: 0 Platforms found");
        		exit(EXIT_FAILURE);
    		}
    		platform = platforms.begin(); //Use first platform found
    		printf("Found Platform: %s\n",(*platform).getInfo<CL_PLATFORM_VENDOR>().c_str());

		//Construct Context
    		cl_context_properties conprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*platform)(), 0};
    		cl::Context context_init( CL_DEVICE_TYPE_GPU, conprops, NULL, NULL, NULL);
            context = context_init;
	
		//Get Devices
    		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
   		if (devices.size() < 1) {
        		printf("ERR: 0 Devices found");
       	 		exit(EXIT_FAILURE);
    		}
    		std::vector<cl::Device>::const_iterator device;
    		for(device=devices.begin(); device != devices.end(); ++device ) {
        		printf("Found Device: %s\n",(*device).getInfo<CL_DEVICE_NAME>().c_str());
    		} 

		//Create Command Queue
    		device = devices.begin(); // Use first device
    		cl::CommandQueue queue_init(context, devices[0], 0, NULL); //Create command queue for 1st device
            queue = queue_init;
    		printf("Created Queue on Device: %s\n",(*device).getInfo<CL_DEVICE_NAME>().c_str()); 
	
		//Build Kernel
    		streamsdk::SDKFile file;
    		if (!file.open(kernel_filename.c_str())) {
        		printf("Source file could not be opened: %s\n", kernel_filename.c_str());
        		exit(EXIT_FAILURE);
    		}

    		cl::Program::Sources source(1, std::make_pair(file.source().data(), file.source().size()));
   	 	cl::Program program = cl::Program(context, source, NULL);
    		err = program.build(devices,"");
    		if (err != CL_SUCCESS) {
			std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]).c_str();
			printf("Build ERR: %s\n", str.c_str());
			//return SDK_FAILURE;
    		}

   	 	cl::Kernel kernel1_init(program, kernel_1024.c_str(), NULL);
   	 	cl::Kernel kernel2_init(program, kernel_1024.c_str(), NULL);
   	 	cl::Kernel kernelTwiddle_init(program, kernel_twiddle.c_str(), NULL);
   	 	cl::Kernel kernel3_init(program, kernel_2.c_str(), NULL);
   	 	cl::Kernel kernel4_init(program, kernel_2.c_str(), NULL);
   	 	cl::Kernel kernelcm_init(program, kernel_cm.c_str(), NULL);
   	 	cl::Kernel kernelilql_init(program, kernel_ilql.c_str(), NULL);
   	 	cl::Kernel kernelis_init(program, kernel_is.c_str(), NULL);

        kernel1 = kernel1_init;
        kernel2 = kernel2_init;
        kernelTwiddle = kernelTwiddle_init;
        kernel3 = kernel3_init;
        kernel4 = kernel4_init;
        kernelcm = kernelcm_init;
        kernelilql = kernelilql_init;
        kernelis = kernelis_init;

    }catch (int error){
		printf("ERROR %d\n", error);
	}

}


float* CorrCL::fft2048(float* in, float* in_i){
    float *inAinit = new float[VSIZE];
    float *inAinit_i = new float[VSIZE];
    float *inBinit = new float[VSIZE];
    float *inBinit_i = new float[VSIZE];

    inAinit = &in[0];
    inAinit_i = &in_i[0];
    inBinit = &in[VSIZE];
    inBinit_i = &in_i[VSIZE];

    try {
		cl_int err;

        float *inA = new float[VSIZE];
        float *inB = new float[VSIZE];
        float *inA_i = new float[VSIZE];
        float *inB_i = new float[VSIZE];
        //Reorder input strings for final 1024x 2 wide FFT
        for (int i=0;i<VSIZE/2;i++) {
            inA[i] = inAinit[2*i];
            inA[i+VSIZE/2] = inBinit[2*i];
            inA_i[i] = inAinit_i[2*i];
            inA_i[i+VSIZE/2] = inBinit_i[2*i];

            inB[i] = inAinit[2*i+1];
            inB[i+VSIZE/2] = inBinit[2*i+1];
            inB_i[i] = inAinit_i[2*i+1];
            inB_i[i+VSIZE/2] = inBinit_i[2*i+1];
        } 

		//Create IO Buffers
		float *outValA = new float[2*VSIZE];
		float *outValB = new float[2*VSIZE];
		Buffer inBufA = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inA[0]);
		Buffer inBufAi = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inA_i[0]);
		Buffer outBufA = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValA[0]);

		Buffer inBufB = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inB[0]);
		Buffer inBufBi = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inB_i[0]);
		Buffer outBufB = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValB[0]);


		//Set Kernel Arguments
		kernel1.setArg(0, inBufA);
		kernel1.setArg(1, inBufAi);
		kernel1.setArg(2, outBufA);	

		kernel2.setArg(0, inBufB);
		kernel2.setArg(1, inBufBi);
		kernel2.setArg(2, outBufB);	

		//Run Kernel
		NDRange global(VSIZE);
		NDRange local(128);
		queue.enqueueNDRangeKernel(kernel1, NullRange, global, local);//, NULL, &event);
		queue.enqueueNDRangeKernel(kernel2, NullRange, global, local);//, NULL, &event);
		
		//Export outBuf to local variable
		float * outputA = (float *) queue.enqueueMapBuffer(outBufA, CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
		float * outputB = (float *) queue.enqueueMapBuffer(outBufB, CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));

        //Reorder and apply Twiddle
		float *outValBTwd = new float[2*VSIZE];
		Buffer fftBufIn2 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &outValB[0]);
		Buffer fftBufIn2i = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &outValB[VSIZE]);
		Buffer fftBufOut2 = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValBTwd[0]);

		kernelTwiddle.setArg(0, fftBufIn2);
		kernelTwiddle.setArg(1, fftBufIn2i);
		kernelTwiddle.setArg(2, fftBufOut2);	
		queue.enqueueNDRangeKernel(kernelTwiddle, NullRange, global, local);
		float * output_twiddle = (float *) queue.enqueueMapBuffer(fftBufOut2, CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));

        float *fftIn1 = new float[VSIZE];
        float *fftIn1i = new float[VSIZE];
        float *fftIn2 = new float[VSIZE];
        float *fftIn2i = new float[VSIZE];

        for (int i=0;i<VSIZE;i++) {
            if (i<VSIZE/2) {
                fftIn1[i] = outValA[i];
                fftIn1i[i] = outValA[i+VSIZE];
                fftIn2[i] = outValA[i+VSIZE/2];
                fftIn2i[i] = outValA[i+3*VSIZE/2];
            } else {
                fftIn1[i] = outValBTwd[i%(VSIZE/2)];
                fftIn1i[i] = outValBTwd[i%(VSIZE/2)+VSIZE];
                fftIn2[i] = outValBTwd[i%(VSIZE/2)+VSIZE/2];
                fftIn2i[i] = outValBTwd[i%(VSIZE/2)+3*VSIZE/2];
            }
        }

		float *outValA2 = new float[2*VSIZE];
		float *outValB2 = new float[2*VSIZE];
		Buffer inBufA2 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn1[0]);
		Buffer inBufA2i = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn1i[0]);
		Buffer outBufA2 = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValA2[0]);

		Buffer inBufB2 = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn2[0]);
		Buffer inBufB2i = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn2i[0]);
		Buffer outBufB2 = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValB2[0]);

		kernel3.setArg(0, inBufA2);
		kernel3.setArg(1, inBufA2i);
		kernel3.setArg(2, outBufA2);	

		kernel4.setArg(0, inBufB2);
		kernel4.setArg(1, inBufB2i);
		kernel4.setArg(2, outBufB2);	
        
		queue.enqueueNDRangeKernel(kernel3, NullRange, global, local);
		queue.enqueueNDRangeKernel(kernel4, NullRange, global, local);

		float * outputA2 = (float *) queue.enqueueMapBuffer(outBufA2, CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
		float * outputB2 = (float *) queue.enqueueMapBuffer(outBufB2, CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));

        //Clean up output buffers
		err = queue.enqueueUnmapMemObject(outBufA, (void *) outputA);
		err = queue.enqueueUnmapMemObject(outBufB, (void *) outputB);
		err = queue.enqueueUnmapMemObject(fftBufOut2, (void *) output_twiddle);
		err = queue.enqueueUnmapMemObject(outBufA2, (void *) outputA);
		err = queue.enqueueUnmapMemObject(outBufB2, (void *) outputB);

        
        float *outAll = new float[4*VSIZE];
        for(int i=0; i<VSIZE/2;i++) {
            outAll[i] = outValA2[i];
            outAll[i+1*VSIZE/2] = outValB2[i];
            outAll[i+2*VSIZE/2] = outValA2[i+VSIZE/2];
            outAll[i+3*VSIZE/2] = outValB2[i+VSIZE/2];
            outAll[i+4*VSIZE/2] = outValA2[i+VSIZE];
            outAll[i+5*VSIZE/2] = outValB2[i+VSIZE];
            outAll[i+6*VSIZE/2] = outValA2[i+3*VSIZE/2];
            outAll[i+7*VSIZE/2] = outValB2[i+3*VSIZE/2];
       } 
    
        return outAll;

    }catch (int error){
        printf("ERROR %d\n", error);
        return 0;
    }
}

float** CorrCL::fft_m2048(float** in_m, float** in_i_m, int len){
    float *inAinit = new float[VSIZE];
    float *inAinit_i = new float[VSIZE];
    float *inBinit = new float[VSIZE];
    float *inBinit_i = new float[VSIZE];


    try {
		cl_int err;

        float *inA = new float[VSIZE];
        float *inB = new float[VSIZE];
        float *inA_i = new float[VSIZE];
        float *inB_i = new float[VSIZE];

        float *in = new float[2*VSIZE];
        float *in_i = new float[2*VSIZE];

        Buffer *inBufA = new Buffer[len];
        Buffer *inBufAi = new Buffer[len];
        Buffer *outBufA = new Buffer[len];

        Buffer *inBufB = new Buffer[len];
        Buffer *inBufBi = new Buffer[len];
        Buffer *outBufB = new Buffer[len];

        Buffer *fftBufIn2 = new Buffer[len];
        Buffer *fftBufIn2i = new Buffer[len];
        Buffer *fftBufOut2 = new Buffer[len];

        Buffer *inBufA2 = new Buffer[len];
        Buffer *inBufA2i = new Buffer[len];
        Buffer *outBufA2 = new Buffer[len];

        Buffer *inBufB2 = new Buffer[len];
        Buffer *inBufB2i = new Buffer[len];
        Buffer *outBufB2 = new Buffer[len];

        float **outValA_m = new float*[len];
        float **outValB_m = new float*[len];
        float **outValBTwd_m = new float*[len];
        float **outValA2_m = new float*[len];
        float **outValB2_m = new float*[len];
        float **outAll_m = new float*[len];

        NDRange global(VSIZE);
        NDRange local(128);

        for (int j=0;j<len;j++) { 
            in = in_m[j];
            in_i = in_i_m[j];

            inAinit = &in[0];
            inAinit_i = &in_i[0];
            inBinit = &in[VSIZE];
            inBinit_i = &in_i[VSIZE];

            //Reorder input strings for final 1024x 2 wide FFT
            for (int i=0;i<VSIZE/2;i++) {
                inA[i] = inAinit[2*i];
                inA[i+VSIZE/2] = inBinit[2*i];
                inA_i[i] = inAinit_i[2*i];
                inA_i[i+VSIZE/2] = inBinit_i[2*i];

                inB[i] = inAinit[2*i+1];
                inB[i+VSIZE/2] = inBinit[2*i+1];
                inB_i[i] = inAinit_i[2*i+1];
                inB_i[i+VSIZE/2] = inBinit_i[2*i+1];
            } 

            //Create IO Buffers
            float *outValA = new float[2*VSIZE];
            float *outValB = new float[2*VSIZE];
            outValA_m[j] = outValA;
            outValB_m[j] = outValB;
            inBufA[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inA[0]);
            inBufAi[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inA_i[0]);
            outBufA[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValA[0]);

            inBufB[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inB[0]);
            inBufBi[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &inB_i[0]);
            outBufB[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValB[0]);


            //Set Kernel Arguments
            kernel1.setArg(0, inBufA[j]);
            kernel1.setArg(1, inBufAi[j]);
            kernel1.setArg(2, outBufA[j]);	

            kernel2.setArg(0, inBufB[j]);
            kernel2.setArg(1, inBufBi[j]);
            kernel2.setArg(2, outBufB[j]);	

            //Run Kernel
            queue.enqueueNDRangeKernel(kernel1, NullRange, global, local);//, NULL, &event);
            queue.enqueueNDRangeKernel(kernel2, NullRange, global, local);//, NULL, &event);
        
        }
	
        for (int j=0;j<len;j++){	
            //Export outBuf to local variable
            float * outputA = (float *) queue.enqueueMapBuffer(outBufA[j], CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
            float * outputB = (float *) queue.enqueueMapBuffer(outBufB[j], CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
        }
        
        for (int j=0;j<len;j++){
            float *outValB = outValB_m[j];
            
            //Reorder and apply Twiddle
            float *outValBTwd = new float[2*VSIZE];
            outValBTwd_m[j] = outValBTwd;
            fftBufIn2[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &outValB[0]);
            fftBufIn2i[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &outValB[VSIZE]);
            fftBufOut2[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValBTwd[0]);

            kernelTwiddle.setArg(0, fftBufIn2[j]);
            kernelTwiddle.setArg(1, fftBufIn2i[j]);
            kernelTwiddle.setArg(2, fftBufOut2[j]);	
            queue.enqueueNDRangeKernel(kernelTwiddle, NullRange, global, local);

        }

        for (int j=0;j<len;j++){
            float * output_twiddle = (float *) queue.enqueueMapBuffer(fftBufOut2[j], CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
        }
        
        for (int j=0;j<len;j++){
            float *fftIn1 = new float[VSIZE];
            float *fftIn1i = new float[VSIZE];
            float *fftIn2 = new float[VSIZE];
            float *fftIn2i = new float[VSIZE];

            float *outValA = outValA_m[j];
            float *outValBTwd = outValBTwd_m[j];

            for (int i=0;i<VSIZE;i++) {
                if (i<VSIZE/2) {
                    fftIn1[i] = outValA[i];
                    fftIn1i[i] = outValA[i+VSIZE];
                    fftIn2[i] = outValA[i+VSIZE/2];
                    fftIn2i[i] = outValA[i+3*VSIZE/2];
                } else {
                    fftIn1[i] = outValBTwd[i%(VSIZE/2)];
                    fftIn1i[i] = outValBTwd[i%(VSIZE/2)+VSIZE];
                    fftIn2[i] = outValBTwd[i%(VSIZE/2)+VSIZE/2];
                    fftIn2i[i] = outValBTwd[i%(VSIZE/2)+3*VSIZE/2];
                }
            }

            float *outValA2 = new float[2*VSIZE];
            float *outValB2 = new float[2*VSIZE];
            outValA2_m[j] = outValA2;
            outValB2_m[j] = outValB2;

            inBufA2[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn1[0]);
            inBufA2i[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn1i[0]);
            outBufA2[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValA2[0]);

            inBufB2[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn2[0]);
            inBufB2i[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, VSIZE * sizeof(float), (void *) &fftIn2i[0]);
            outBufB2[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &outValB2[0]);

            kernel3.setArg(0, inBufA2[j]);
            kernel3.setArg(1, inBufA2i[j]);
            kernel3.setArg(2, outBufA2[j]);	

            kernel4.setArg(0, inBufB2[j]);
            kernel4.setArg(1, inBufB2i[j]);
            kernel4.setArg(2, outBufB2[j]);	
            
            queue.enqueueNDRangeKernel(kernel3, NullRange, global, local);
            queue.enqueueNDRangeKernel(kernel4, NullRange, global, local);
        }
    
        for (int j=0;j<len;j++){

            float * outputA2 = (float *) queue.enqueueMapBuffer(outBufA2[j], CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
            float * outputB2 = (float *) queue.enqueueMapBuffer(outBufB2[j], CL_TRUE, CL_MAP_READ, 0, 2*VSIZE * sizeof(float));
        }
        for (int j=0;j<len;j++){
            //Clean up output buffers
//            err = queue.enqueueUnmapMemObject(outBufA, (void *) outputA);
//            err = queue.enqueueUnmapMemObject(outBufB, (void *) outputB);
//            err = queue.enqueueUnmapMemObject(fftBufOut2, (void *) output_twiddle);
//            err = queue.enqueueUnmapMemObject(outBufA2, (void *) outputA);
//            err = queue.enqueueUnmapMemObject(outBufB2, (void *) outputB);

            float *outValA2 = outValA2_m[j];
            float *outValB2 = outValB2_m[j];
            
            float *outAll = new float[4*VSIZE];
            outAll_m[j] = outAll;
            for(int i=0; i<VSIZE/2;i++) {
                outAll[i] = outValA2[i];
                outAll[i+1*VSIZE/2] = outValB2[i];
                outAll[i+2*VSIZE/2] = outValA2[i+VSIZE/2];
                outAll[i+3*VSIZE/2] = outValB2[i+VSIZE/2];
                outAll[i+4*VSIZE/2] = outValA2[i+VSIZE];
                outAll[i+5*VSIZE/2] = outValB2[i+VSIZE];
                outAll[i+6*VSIZE/2] = outValA2[i+3*VSIZE/2];
                outAll[i+7*VSIZE/2] = outValB2[i+3*VSIZE/2];
           } 
        }
    
        return outAll_m;

    }catch (int error){
        printf("ERROR %d\n", error);
        return 0;
    }
}

float* CorrCL::ifft2048(float* inAinit){

    float *inA = new float[2*VSIZE];
    float *inA_i = new float[2*VSIZE];

    inA = &inAinit[0];
    inA_i = &inAinit[2*VSIZE];

    float *fft_out = new float[4*VSIZE];
    fft_out = fft2048(inA, inA_i);
    
    //ifft scale
    try {
		cl_int err;
   
        //Create IO Buffers
		float *outVal = new float[4*VSIZE];
		Buffer inBuf = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &fft_out[0]);
		Buffer outBuf = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);

		//Set Kernel Arguments
		kernelis.setArg(0, inBuf);
		kernelis.setArg(1, outBuf);	

		//Run Kernel
		NDRange global(4*VSIZE);
		NDRange local(VSIZE);
		queue.enqueueNDRangeKernel(kernelis, NullRange, global, NullRange);

		//Export outBuf to local variable
		float * output = (float *) queue.enqueueMapBuffer(outBuf, CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));

        return outVal;

    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

    
}

float** CorrCL::ifft_m2048(float** inAinit_m, int len){

    float **inA_m = new float*[len];
    float **inA_i_m = new float*[len];

    for (int j=0;j<len;j++){
        float* inAinit = inAinit_m[j];
        float *inA = new float[2*VSIZE];
        float *inA_i = new float[2*VSIZE];

        inA = &inAinit[0];
        inA_i = &inAinit[2*VSIZE];

        inA_m[j] = inA;
        inA_i_m[j] = inA_i;
    }

    float **fft_out_m = new float*[len];
    fft_out_m = fft_m2048(inA_m, inA_i_m, len);
    
    //ifft scale
    try {
		cl_int err;
        float **outVal_m = new float*[len];
        Buffer *inBuf = new Buffer[len];
        Buffer *outBuf = new Buffer[len];

        NDRange global(4*VSIZE);
        NDRange local(VSIZE);

        for (int j=0;j<len;j++){ 
            //Create IO Buffers
            float *fft_out = fft_out_m[j];
            float *outVal = new float[4*VSIZE];
            outVal_m[j] = outVal;
            inBuf[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &fft_out[0]);
            outBuf[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);

            //Set Kernel Arguments
            kernelis.setArg(0, inBuf[j]);
            kernelis.setArg(1, outBuf[j]);	

            //Run Kernel
            queue.enqueueNDRangeKernel(kernelis, NullRange, global, NullRange);
        }

        for (int j=0;j<len;j++){
            //Export outBuf to local variable
            float * output = (float *) queue.enqueueMapBuffer(outBuf[j], CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));
        }
        return outVal_m;

    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

    
}

float* CorrCL::cm2048(float* inA, float* inB){
    try {
		cl_int err;
   
        //Create IO Buffers
		float *outVal = new float[4*VSIZE];
		Buffer inBufA = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &inA[0]);
		Buffer inBufB = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &inB[0]);
		Buffer outBuf = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);
 

		//Set Kernel Arguments
		kernelcm.setArg(0, inBufA);
		kernelcm.setArg(1, inBufB);
		kernelcm.setArg(2, outBuf);	

		//Run Kernel
		NDRange global(4*VSIZE);
		NDRange local(2*VSIZE);
		queue.enqueueNDRangeKernel(kernelcm, NullRange, global, NullRange);

		//Export outBuf to local variable
		float * output = (float *) queue.enqueueMapBuffer(outBuf, CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));

        return outVal;
    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

}


float** CorrCL::cm_m2048(float** inA_m, float* inB, int len){
    try {
		cl_int err;
        Buffer *inBufA = new Buffer[len];        
        Buffer *inBufB = new Buffer[len];
        Buffer *outBuf = new Buffer[len];
        float **outVal_m = new float*[len];
        
        NDRange global(4*VSIZE);
        NDRange local(2*VSIZE);

        for (int j=0;j<len;j++) { 
            float* inA = inA_m[j];
            //Create IO Buffers
            float *outVal = new float[4*VSIZE];
            outVal_m[j] = outVal;
            inBufA[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &inA[0]);
            inBufB[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &inB[0]);
            outBuf[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);
     
            //Set Kernel Arguments
            kernelcm.setArg(0, inBufA[j]);
            kernelcm.setArg(1, inBufB[j]);
            kernelcm.setArg(2, outBuf[j]);	

            //Run Kernel
            queue.enqueueNDRangeKernel(kernelcm, NullRange, global, NullRange);
        }
        
        for (int j=0;j<len;j++){
            //Export outBuf to local variable
            float * output = (float *) queue.enqueueMapBuffer(outBuf[j], CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));
        }
        return outVal_m;
    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

}


float* CorrCL::ilql2048(float* inPrn, float* inPrnTime, float freq){
    try {
		cl_int err;
   
        //Create IO Buffers
		float *outVal = new float[4*VSIZE];
		Buffer inBufA = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &inPrn[0]);
		Buffer inBufB = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &inPrnTime[0]);
		Buffer outBuf = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);
 

		//Set Kernel Arguments
		kernelilql.setArg(0, inBufA);
		kernelilql.setArg(1, inBufB);
		kernelilql.setArg(2, freq);
		kernelilql.setArg(3, outBuf);	

		//Run Kernel
		NDRange global(4*VSIZE);
		NDRange local(2*VSIZE);
		queue.enqueueNDRangeKernel(kernelilql, NullRange, global, NullRange);

		//Export outBuf to local variable
		float * output = (float *) queue.enqueueMapBuffer(outBuf, CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));

        return outVal;
    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

}


float** CorrCL::ilql_m2048(float* inPrn, float* inPrnTime, float* d_v, float freq, int len){
    float **IlQl_m = new float*[len];
    float inFreq=0;
    Buffer *inBufA = new Buffer[len];
    Buffer *inBufB = new Buffer[len];
    Buffer *outBuf = new Buffer[len];
    NDRange global(4*VSIZE);
    NDRange local(2*VSIZE);

    try {
		cl_int err;
		float *outVal = new float[4*VSIZE];
        
        for (int j=0;j<len;j++) {

            float *outVal = new float[4*VSIZE];
            IlQl_m[j] = outVal;
            inFreq = freq + d_v[j];
            //Create IO Buffers
            inBufA[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &inPrn[0]);
            inBufB[j] = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2*VSIZE * sizeof(float), (void *) &inPrnTime[0]);
            outBuf[j] = Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 4*VSIZE * sizeof(float), (void *) &outVal[0]);
     

            //Set Kernel Arguments
            kernelilql.setArg(0, inBufA[j]);
            kernelilql.setArg(1, inBufB[j]);
            kernelilql.setArg(2, inFreq);
            kernelilql.setArg(3, outBuf[j]);	

            //Run Kernel
            queue.enqueueNDRangeKernel(kernelilql, NullRange, global, NullRange);

        }
        for (int j=0;j<len;j++) {
		    //Export outBuf to local variable
		    float * output = (float *) queue.enqueueMapBuffer(outBuf[j], CL_TRUE, CL_MAP_READ, 0, 4*VSIZE * sizeof(float));
        }
        return IlQl_m;
    } catch(int error){
        printf("ERROR %d\n", error);
        return 0;
    }

}

