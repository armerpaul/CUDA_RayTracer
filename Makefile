CPP = g++
GFLAGS = -g
NVCC = nvcc
NVFLAGS = -gencode arch=compute_20,code=sm_20 -O3
LIBS = -lcudart
all: gpu
gpu: Image.cpp cudaRayTrace.cu cudaRayTrace.h types.h
	$(NVCC) $(NVFLAGS) -o gpu Image.cpp cudaRayTrace.cu $(LIBS)
cpu: Image.cpp main.cpp VanExLib.h types.h
	g++ -o cpu -g Image.cpp main.cpp 
clean:
	rm gpu *.tga
