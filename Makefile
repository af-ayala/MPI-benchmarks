#!/bin/bash

CCFLAGS  =	-g -O3 -std=c++11 #-pg
CC       =  mpicxx $(CCFLAGS)  #-fopenmp

# Select type of allocation memory, either on CPU (default) or GPU, for the latter run as:  'make mem=GPU'
mem     =
MEM		  =	$(shell echo $(mem) | tr a-z A-Z)
MEM_DEF = -DMEM_$(MEM)

ifneq (,$(filter GPU GPU_MAN GPU_REG,$(MEM)))
CUDA_INC=-I$(CUDADIR)/include
CUDA_LIB=-L$(CUDADIR)/lib64 -lcuda -lcudart
endif

INC= $(MEM_INC) $(MEM_DEF) $(CUDA_INC)
LIB= $(CUDA_LIB)

all : mpi_bw

mpi_bw: mpi_bw.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LIB)

%.o:%.cpp
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

clean:
	@rm -f *.o mpi_bw
