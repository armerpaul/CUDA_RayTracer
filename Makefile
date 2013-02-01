CPP = g++
NVCC = nvcc
NVFLAGS = -v -O2

all: Image.cpp main.cu
        $(NVCC) $(NVGLAGS) Image.cpp main.cu