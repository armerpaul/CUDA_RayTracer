CPP = g++
GFLAGS = -g
NVCC = nvcc
NVFLAGS = -g -G -gencode arch=compute_20,code=sm_20
LIBS = -lcudart
all: Image.cpp main.cu
	$(NVCC) $(NVFLAGS) Image.cpp main.cu $(LIBS)
cpu: Image.cpp main.cpp
	g++ -g Image.cpp main.cpp 
