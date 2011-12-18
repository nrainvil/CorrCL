DEPTH = C:/Users/nrainvil/Documents/AMDAPP

include $(DEPTH)/make/openclsdkdefs.mk 

####
#
#  Targets
#
####

OPENCL			= 1
SAMPLE_EXE		= 1
EXE_TARGET 		= CorrTest
EXE_TARGET_INSTALL   	= CorrTest

####
#
#  C/CPP files
#
####

FILES 	= CorrCL CorrTest 
CLFILES = CorrCL_kernels.cl 

LLIBS  	+= SDKUtil

include $(DEPTH)/make/openclsdkrules.mk 

