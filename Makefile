export CC  = gcc
export CXX = g++
export NVCC =nvcc
export DEBUG= -ggdb  -O3
export NVCCDEBUG= -g -O3

export CFLAGS = -Wall   -msse3 -Wno-unknown-pragmas -funroll-loops -I./mshadow/ -I/usr/include/python2.7 -L ../opt/lib

LDFLAGS= -lprotobuf -lglog -lopenblas -fopenmp -lm -lcudart -lcublas -lcurand -lleveldb  -lpython2.7 -lz `pkg-config --libs opencv` 
CFLAGS+= -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_CBLAS=1

export NVCCFLAGS = --use_fast_math -ccbin $(CXX)

BIN = 
OBJ =  nnet_cpu.o data.o
CUOBJ =  nnet_gpu.o
CUBIN =  bin/net
PROTO = layernet.pb.cc  layernet.pb.h  layernet_pb2.py
.PHONY: clean all

all: $(PROTO) $(BIN) $(OBJ) $(CUBIN) $(CUOBJ) 


$(PROTO):
	protoc -I layernet/proto --cpp_out=. --python_out=. layernet/proto/layernet.proto 
	-mkdir saves
	-mkdir bin
	
nnet_gpu.o: layernet/nnet/nnet.cu  layernet/core/*.hpp layernet/core/*.h  layernet/nnet/*.hpp layernet/nnet/*.h
nnet_cpu.o: layernet/nnet/nnet.cpp layernet/core/*.hpp layernet/core/*.h layernet/nnet/*.hpp layernet/nnet/*.h
data.o: layernet/io/data.cpp  layernet/io/data.h layernet/io/*.hpp  
bin/net:layernet/net_main.cpp  layernet.pb.cc data.o nnet_cpu.o nnet_gpu.o 

$(BIN) :
	$(CXX) $(CFLAGS) $(DEBUG)  -o $@ $(filter %.cpp %.o %.c, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) $(DEBUG) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS) $(DEBUG)" $(filter %.cu, $^)
$(CUBIN) :
	$(NVCC) -o $@ $(NVCCDEBUG) $(NVCCFLAGS)  -Xcompiler "$(CFLAGS) $(DEBUG)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o %.cc, $^)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~  $(PROTO)


