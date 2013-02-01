CPP = g++
NVCC = nvcc
NVFLAGS = -g -gencode arch=compute_20,code=sm_20
LIBS = -lcudart
all: Image.cpp main.cu
	$(NVCC) $(NVFLAGS) Image.cpp main.cu $(LIBS)
