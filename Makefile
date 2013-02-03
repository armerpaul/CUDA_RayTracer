CPP = g++
GFLAGS = -g
NVCC = nvcc
NVFLAGS = -g -G -gencode arch=compute_20,code=sm_20
LIBS = -lcudart
all: cpu gpu
gpu: Image.cpp main.cu VanExLib.h types.h
	$(NVCC) $(NVFLAGS) -o gpu Image.cpp main.cu $(LIBS)
cpu: Image.cpp main.cpp VanExLib.h types.h
	g++ -o cpu -g Image.cpp main.cpp 
clean:
	rm gpu cpu
