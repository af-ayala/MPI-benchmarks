#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>

#if defined(MEM_GPU) || defined(MEM_GPU_MAN) || defined(MEM_GPU_REG)
#include "cuda.h"
#include "cuda_runtime.h"
#endif

//typedef char SCALAR;
//#define MPI_SCALAR MPI_CHAR
typedef double SCALAR;
#define MPI_SCALAR MPI_DOUBLE

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif



void sfree(void *ptr);
void *smalloc(int64_t nbytes);

int my_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

// xprintf: includes the buffer flushing and node rank
#define xprintf(...) { \
    fprintf( stdout, "[node %i]: ", my_rank()); \
    fprintf( stdout,  __VA_ARGS__ ) ;          \
    fflush(  stdout );                          \
}

#define MPI_CHECK(stmt)                                          \
do {                                                             \
   int mpi_errno = (stmt);                                       \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
} while (0)


#define magma_check_cuda_error() do { \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(e); \
 } \
} while(0)


#if defined(MEM_GPU) || defined(MEM_GPU_MAN) || defined(MEM_GPU_REG)

void magma_Alltoallv(const void *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm){

    int me, nprocs, j=0;
    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nprocs);
    MPI_Request request[nprocs];

    if (comm != MPI_COMM_NULL) {

        for(int neighbor=0; neighbor<nprocs; neighbor++){
              if(me!=neighbor){
                MPI_Irecv(recvbuf+rdispls[neighbor], recvcounts[neighbor], sendtype, neighbor, 0, comm, request + j);
                j++;
              }
        }

        for(int neighbor=0; neighbor<nprocs; neighbor++){
              if(me!=neighbor){
                MPI_Isend(sendbuf+sdispls[neighbor], sendcounts[neighbor], sendtype, neighbor, 0, comm, request + j);
                j++;
              }
              else{
                cudaMemcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype),  cudaMemcpyDeviceToDevice);
              }
        }

        MPI_Waitall(j, request, MPI_STATUSES_IGNORE);
    }
}

#endif
