CPP = g++
NVCC = nvcc
NVFLAGS = -v -O2 -gencode arch=compute_20,code=sm_20

all: Image.cpp main.cu
	$(NVCC) $(NVFLAGS) Image.cpp main.cu
