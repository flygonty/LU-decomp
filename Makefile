NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_90
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcublas -lcusolver -lcudart
EXE	        = lu-fact
OBJ	        = main.o

default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
