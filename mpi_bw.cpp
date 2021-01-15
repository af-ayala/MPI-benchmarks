/**
 * Benchmark of MPI routines on CPUs and GPUs
 * Author: Alan Ayala
 * Innovative Computing Laboratory, University of Tennessee at Knoxville
 * Last mod: 3/22/2019
 * For help: ./mpi_bw -h
*/

#include "tools.h"
///////////////////////////////////////////////////////////////
static void
show_help(char *progname) {
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( " Compilation             \n");
  printf( " make clean && make              CPU data (malloc)      \n");
  printf( " make clean && make mem=GPU      GPU data (cudaMalloc)  \n");
  printf( " make clean && make mem=GPU_MAN  CPU/GPU data (cudaMallocManaged) \n");
  printf( " make clean && make mem=GPU_REG  CPU/GPU data (malloc + cudaHostRegister)\n");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( "  \n");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( "%s  USAGE OPTIONS:  ./%s MPI_FUNC_NAME  size  num_iterations \n", progname, progname );
  printf( "  MPI_FUNC_NAME can be: )\n");
  printf( "                        blocking MPI calls\n");
  printf( "                        SEND       works only for np=2\n");
  printf( "                        ALL2ALLV \n");
  printf( "                        MAGMA_A2AV \n");
  printf( "                        BCAST \n");
  printf( "                        GATHER \n");
  printf( "                        GATHERV \n");
  printf( "                        SCATTER \n");
  printf( "                        SCATTERV \n");
  printf( "                        ALLGATHER \n");
  printf( "                        ALLREDUCE \n");
  printf( "                        \n");
  printf( "                        Non blocking MPI calls\n");
  printf( "                        SEND_B2B   work only for np=2\n");
  printf( "                        SEND_BIDIR work only for np=2\n");
  printf( "                        \n");
  printf( "  size is the number of double \n");
  printf( "  num_iterations is the number of iteration the MPI_operation is going to be performed \n");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( "  \n");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( " Run examples - OpenMPI \n");
  printf( " mpirun -n 2 ./mpi_bw  SEND         2000000      3 \n");
  printf( " mpirun -n 4 ./mpi_bw  ALLGATHER    2000000      3 \n");
  printf( " mpirun -n 4 ./mpi_bw  ALL2ALLV     2000000      3 \n");
  printf( " mpirun -n 4 ./mpi_bw  ALLREDUCE    2000000      3 \n");
  printf( " mpirun -n 2 ./mpi_bw  SEND_B2B     2000000      3 \n");
  printf( " mpirun -n 2 ./mpi_bw  SEND_BIDIR   2000000      3 \n");
  printf( " mpirun -n 2 --map-by node nvprof -fo profiler_output_%s ./mpi_bw SEND 2000000  1\n","%q{OMPI_COMM_WORLD_RANK}.nvvp");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( " Run examples - SpectrumMPI \n");
  printf( " jsrun --smpiargs=\"-gpu\" -r1 -a1 -c1 -g1 -n2 ./gpu_setter_summit.sh ./mpi_bw SEND   2000000 4 \n");
  printf( " jsrun --smpiargs=\"-gpu\" -a1 -c1 -g1 -n2 ./gpu_setter_summit.sh ./mpi_bw SEND_BIDIR 2000000 4 \n");
  printf( " jsrun --smpiargs=\"-gpu\" -a1 -c1 -g1 -n2 ./gpu_setter_summit.sh ./mpi_bw ALL2ALLV   2000000 4 \n");
  printf( " jsrun --smpiargs=\"-gpu\" -a1 -c1 -g1 -n2 nvprof -o $MEMBERWORK/csc301/bw.%%p.nvvp ./gpu_setter_summit.sh ./mpi_bw SEND 2000000 4\n");
  printf( " jsrun --smpiargs=\"-gpu\" -a1 -c1 -g1 -n2 nvprof -o $MEMBERWORK/csc301/bw.%%p.nvvp ./gpu_setter_summit.sh ./mpi_bw ALL2ALLV 2000000 4\n");
  printf( " Get profiler without gpu_setter (use cuda 9.1.85 to be able to see it using NVVP): \n");
  printf( " jsrun --smpiargs=\"-gpu  -x CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'\" -a1 -c1 -g1 -n3 nvprof -o $MEMBERWORK/csc301/bw.%%p.nvvp ./mpi_bw ALL2ALLV 2000000 1\n");
  printf( " ---------------------------------------------------------------------------------------------- \n");
  printf( "  \n");
}
///////////////////////////////////////////////////////////////
//
int main(int argc, char *argv[]) {
    int nprocs, myrank;
    int i, j, size;

    int num_iterations = 1;  // 1 iteration by default

    MPI_Status status;
    double t=0.0, t_start=0.0, t_end=0.0, iter_time = 0.0, max_iter_time=0.0, min_iter_time=1e5;
    SCALAR aux = 0.0;
    double bw = 0.0, min_bw=0.0, max_bw=0.0;
    char option_bench[30]; // By default we measure point2point bandwidth


    if(argc < 2) {
      show_help( argv[0] );
      return 0;
    }
    for( int i = 1; i < argc; ++i ) {
        if( (strcmp("--help", argv[i]) == 0) || (strcmp("-h", argv[i]) == 0) ){
            show_help( argv[0] );
            return 0;
        }
    }


    MPI_CHECK(MPI_Init(&argc,&argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD,&nprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD,&myrank));

    strcpy(option_bench,argv[1]);
    size = atoi(argv[2]);  // message size to send by each processor
    num_iterations = atoi(argv[3]);

// 1. Initialize
    if (myrank == 0){
      printf("=========================================\n");
      printf("    Benchmark of routine MPI_%s \n",option_bench);
      printf("=========================================\n");
      printf("Total Number of processes : %i\n",nprocs);
      printf("Message size = %d \n", size);
      printf("Number of iterations = %d \n", num_iterations);
    }

    char hostname[64];
    gethostname(hostname, 64);
    xprintf("Node: %s \n", hostname);

// Print available devices and assign to each process
  #if defined(MEM_GPU) || defined(MEM_GPU_MAN) || defined(MEM_GPU_REG)
    int ndevices = 0;
    cudaGetDeviceCount( &ndevices );
    magma_check_cuda_error();
    xprintf("has %d GPU's in total\n", ndevices);
    int gpu_dev = myrank%ndevices;

    #if 1
    cudaSetDevice(gpu_dev);
    #else
    cudaGetDevice(&gpu_dev);
    #endif

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, gpu_dev );
    magma_check_cuda_error();
    xprintf( "Selected device = %d: %s, %.1f MHz clock, %.1f MiB memory, capability %d.%d\n",
      gpu_dev, prop.name, prop.clockRate / 1000.,  prop.totalGlobalMem / (1024.*1024.), prop.major, prop.minor );
  #endif

// Warming up
///////////////////////////////////////////////////////////////
  SCALAR *toto, *toto2;
  int cnt=32;
  toto = (SCALAR *) smalloc(cnt*sizeof(SCALAR));
  toto2 = (SCALAR *) smalloc(cnt*sizeof(SCALAR));
  //if(myrank==0) MPI_CHECK(MPI_Send(toto, 32, MPI_SCALAR, 1, 123, MPI_COMM_WORLD));
  //if(myrank==1) MPI_CHECK(MPI_Recv(toto, 32, MPI_SCALAR, 0, 123, MPI_COMM_WORLD, &status));
  //if(myrank==0) MPI_Sendrecv(toto, cnt, MPI_SCALAR, 0, 123, toto2, cnt, MPI_SCALAR, 0, 123, MPI_COMM_WORLD, &status);
   MPI_Bcast(toto, cnt, MPI_SCALAR, 0, MPI_COMM_WORLD);
///////////////////////////////////////////////////////////////

// 2. Allocate buffers,
  SCALAR *buffer_s;
  SCALAR *buffer_r;
  int64_t nbytes;
  nbytes = ((int64_t) sizeof(SCALAR)) * size;
  if(!strcmp("ALL2ALLV",option_bench)  || !strcmp("SCATTER",option_bench) || !strcmp("SCATTERV",option_bench))  nbytes = nprocs*nbytes;
  if(!strcmp("ALLGATHER",option_bench) || !strcmp("GATHER",option_bench)  || !strcmp("GATHERV",option_bench))   nbytes = nprocs*nbytes;
  if(!strcmp("MAGMA_A2AV",option_bench) ) nbytes = nprocs*nbytes; 

  //printf ("voici nbytes %ld \n", nbytes);
  buffer_s = (SCALAR *) smalloc(nbytes);
  buffer_r = (SCALAR *) smalloc(nbytes);
  if (nbytes && buffer_s == NULL) xprintf("buffer_s allocation failed \n");
  if (nbytes && buffer_r == NULL) xprintf("buffer_r allocation failed \n");

// 2.1 Print available devices and assign to each process
  #if defined(MEM_GPU) || defined(MEM_GPU_MAN) || defined(MEM_GPU_REG)
    cudaMemset(buffer_s,0,nbytes);
    cudaMemset(buffer_r,0,nbytes);
  #else
    memset(buffer_s,0,nbytes);
    memset(buffer_r,0,nbytes);
  #endif

// 3. Benchmark MPI routine
  MPI_Barrier(MPI_COMM_WORLD);

// =============
// MPI_Alltoallv
// =============
  if(!strcmp("ALL2ALLV",option_bench)){
      int *rdispls=NULL, *recvcounts=NULL, *sdispls=NULL, *sendcounts=NULL;
      rdispls = new int[nprocs]; recvcounts = new int[nprocs];  sdispls = new int[nprocs];  sendcounts = new int[nprocs];
      int disp = 0;
      for ( i = 0; i < nprocs; i++) {
          recvcounts[i] = size;
          sendcounts[i] = size;
          rdispls[i] = disp;
          sdispls[i] = disp;
          disp += size;
      }
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Alltoallv(buffer_s, sendcounts, sdispls, MPI_SCALAR, buffer_r, recvcounts, rdispls, MPI_SCALAR, MPI_COMM_WORLD));
          // sleep(0.01);
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start) / ( (double) num_iterations) / ( (double) nprocs );

      delete [] rdispls;
      delete [] recvcounts;
      delete [] sdispls;
      delete [] sendcounts;
  }


// ===============
// magma_Alltoallv
// ===============
#if defined(MEM_GPU) || defined(MEM_GPU_MAN) || defined(MEM_GPU_REG)

  else if(!strcmp("MAGMA_A2AV",option_bench)){
      int *rdispls=NULL, *recvcounts=NULL, *sdispls=NULL, *sendcounts=NULL;
      rdispls = new int[nprocs]; recvcounts = new int[nprocs];  sdispls = new int[nprocs];  sendcounts = new int[nprocs];
      int disp = 0;
      for ( i = 0; i < nprocs; i++) {
          recvcounts[i] = size;
          sendcounts[i] = size;
          rdispls[i] = disp;
          sdispls[i] = disp;
          disp += size;
      }
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          magma_Alltoallv(buffer_s, sendcounts, sdispls, MPI_SCALAR, buffer_r, recvcounts, rdispls, MPI_SCALAR, MPI_COMM_WORLD);
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start) / ( (double) num_iterations) / ( (double) nprocs );

      delete [] rdispls;
      delete [] recvcounts;
      delete [] sdispls;
      delete [] sendcounts;
  }

#endif

// =========
// MPI_Bcast
// =========
  else if(!strcmp("BCAST",option_bench)){
    int root = 0;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Bcast(buffer_s, size, MPI_SCALAR, root, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }



// ===============================================================================================
// MPI_Scatter
// Note: MPI_Bcast sends the same piece of data to all processes while MPI_Scatter sends chunks of
// an array to different processes.
// ===============================================================================================
  else if(!strcmp("SCATTER",option_bench)){
    int root = 0;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Scatter(buffer_s, size, MPI_SCALAR, buffer_r, size, MPI_SCALAR, root, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }

// ===============================================================================================
// MPI_Gather
// ===============================================================================================
  else if(!strcmp("GATHER",option_bench)){
    int root = 0;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Gather(buffer_s, size, MPI_SCALAR, buffer_r, size, MPI_SCALAR, root, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }



// ==================================================================================
// MPI_Scatterv
// Allows the sending sizes to vary, as well as the pointer to their sending location
// ==================================================================================
  else if(!strcmp("SCATTERV",option_bench)){
    int *sdispls=NULL, *sendcounts=NULL;
    sdispls = new int[nprocs]; sendcounts = new int[nprocs];
    int disp = 0;
    for ( i = 0; i < nprocs; i++) {
        sendcounts[i] = size;
        sdispls[i] = disp;
        disp += size;
    }

    int root = 0;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Scatterv(buffer_s,  sendcounts, sdispls, MPI_SCALAR, buffer_r, size, MPI_SCALAR, root, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }



// ======================================================================================
// MPI_Gatherv
// Allows the receiving sizes to vary, as well as the pointer to their receiving location
// ======================================================================================
  else if(!strcmp("GATHERV",option_bench)){
    int *rdispls=NULL, *recvcounts=NULL;
    rdispls = new int[nprocs]; recvcounts = new int[nprocs];
    int disp = 0;
    for ( i = 0; i < nprocs; i++) {
        recvcounts[i] = size;
        rdispls[i] = disp;
        disp += size;
    }

    int root = 0;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Gatherv(buffer_s, size, MPI_SCALAR, buffer_r, recvcounts, rdispls, MPI_SCALAR, root, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }



// =================================================================================
// MPI_Allgather
// In the most basic sense, MPI_Allgather is an MPI_Gather followed by an MPI_Bcast.
// =================================================================================
  else if(!strcmp("ALLGATHER",option_bench)){
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Allgather(buffer_s, size, MPI_SCALAR, buffer_r, size, MPI_SCALAR, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start)/( (double) num_iterations);
  }

// =============
// MPI_Allreduce
// =============
  else if(!strcmp("ALLREDUCE",option_bench)){
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Allreduce(buffer_s, buffer_r, size, MPI_SCALAR, MPI_SUM, MPI_COMM_WORLD));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end=MPI_Wtime();
      t = (t_end - t_start) / ( (double) num_iterations);
  }

// ==============================================
// Point to point bandwidth, only for 2 processor
// Routines MPI_Send and MPI_Recv
// ==============================================
  else if(!strcmp("SEND",option_bench)){

    if(nprocs != 2) {
        if(myrank == 0)  fprintf(stderr, "ERROR: this test requires exactly two processes!!\n");
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if(myrank == 0) {
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
          iter_time = MPI_Wtime();
          MPI_CHECK(MPI_Send(buffer_s, size, MPI_SCALAR, 1, i, MPI_COMM_WORLD));
          MPI_CHECK(MPI_Recv(buffer_r, size, MPI_SCALAR, 1, i, MPI_COMM_WORLD, &status));
          iter_time = MPI_Wtime() - iter_time;
          max_iter_time = max(max_iter_time, iter_time);
          min_iter_time = min(min_iter_time, iter_time);
      }
      t_end = MPI_Wtime();
      t = (t_end - t_start) / 2.0 / ((double) num_iterations);
      max_iter_time = max_iter_time / 2.0;
      min_iter_time = min_iter_time / 2.0;
    }
    else if(myrank == 1) {
      for(i = 0; i < num_iterations; i++) {
          MPI_CHECK(MPI_Recv(buffer_r, size, MPI_SCALAR, 0, i, MPI_COMM_WORLD, &status));
          MPI_CHECK(MPI_Send(buffer_s, size, MPI_SCALAR, 0, i, MPI_COMM_WORLD));
      }
    }
  }

// ===============================================
// Point to point bandwidth - Non-blocking version
// ===============================================
    else if(!strcmp("NB_SEND_RECV",option_bench)){

      double tot_time = 0.0;
      if(nprocs != 2) {
          if(myrank == 0)  fprintf(stderr, "ERROR: this test requires exactly two processes!!\n");
          MPI_CHECK(MPI_Finalize());
          exit(EXIT_FAILURE);
      }
      MPI_Request req[2];

      int dest = (myrank+1)%2;
      t_start = MPI_Wtime();
      for(i = 0; i < num_iterations; i++) {
        MPI_CHECK(MPI_Isend(buffer_s, size, MPI_SCALAR, dest, 1, MPI_COMM_WORLD, &req[0]));
        MPI_CHECK(MPI_Irecv(buffer_r, size, MPI_SCALAR, dest, 1, MPI_COMM_WORLD, &req[1]));
      }
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      t_end = MPI_Wtime();
      t = (t_end - t_start) / 2.0 / ((double) num_iterations);
    }

// =========================================================================================================
// Point to point bandwidth
// Send back-to-back messages to find maximum sustained date rate that can be achieved at the network level.
// Analogous to the one from OSU benchmark library.
// =========================================================================================================
    else if(!strcmp("SEND_B2B",option_bench)){
      MPI_Request request[100];
      MPI_Status  reqstat[100];

      int num_messages_sent = 1;
      int num_warmup = 0;

      if(nprocs != 2) {
          if(myrank == 0)  fprintf(stderr, "ERROR: this test requires exactly two processes!!\n");
          MPI_CHECK(MPI_Finalize());
          exit(EXIT_FAILURE);
      }

      if(myrank == 0) {
          t_start = MPI_Wtime();
          for(i = 0; i < num_iterations + num_warmup; i++) {
            for(j = 0; j < num_messages_sent; j++) {
                MPI_CHECK(MPI_Isend(buffer_s, size, MPI_SCALAR, 1, 100, MPI_COMM_WORLD, request + j));
            }
            MPI_CHECK(MPI_Waitall(num_messages_sent, request, reqstat));
            MPI_CHECK(MPI_Recv(buffer_r, 1, MPI_SCALAR, 1, 101, MPI_COMM_WORLD, &reqstat[0]));
        }
        t_end = MPI_Wtime();
        t = (t_end - t_start) / ((double) num_iterations*num_messages_sent);
      }
      else if(myrank == 1) {

        for(i = 0; i < num_iterations + num_warmup; i++) {
            for(j = 0; j < num_messages_sent; j++) {
                MPI_CHECK(MPI_Irecv(buffer_r, size, MPI_SCALAR, 0, 100, MPI_COMM_WORLD, request + j));
            }
            MPI_CHECK(MPI_Waitall(num_messages_sent, request, reqstat));
            MPI_CHECK(MPI_Send(buffer_s, 1, MPI_SCALAR, 0, 101, MPI_COMM_WORLD));
        }
      }
    }

// =======================================
// Point to point bi-directional bandwidth
// =======================================
      else if(!strcmp("SEND_BIDIR",option_bench)){

        MPI_Request send_request[100];
        MPI_Request recv_request[100];
        MPI_Request request[100];
        MPI_Status  reqstat[100];

        int i,j;
        int num_messages_sent = 1;
        int num_warmup = 0;

        if(nprocs != 2) {
            if(myrank == 0)  fprintf(stderr, "ERROR: this test requires exactly two processes!!\n");
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        }

        if(myrank == 0) {
          for(i = 0; i < num_iterations + num_warmup; i++) {
              // if(i == num_warmup){
                  t_start = MPI_Wtime();
                // }

              for(j = 0; j < num_messages_sent; j++) {
              MPI_CHECK(MPI_Irecv(buffer_r, size, MPI_SCALAR, 1, 10, MPI_COMM_WORLD, recv_request + j));
              }

              for(j = 0; j < num_messages_sent; j++) {
              MPI_CHECK(MPI_Isend(buffer_s, size, MPI_SCALAR, 1, 100, MPI_COMM_WORLD, send_request + j));
              }

              MPI_CHECK(MPI_Waitall(num_messages_sent, send_request, reqstat));
              MPI_CHECK(MPI_Waitall(num_messages_sent, recv_request, reqstat));
          }
          t_end = MPI_Wtime();
          // t = (t_end - t_start) / ((double) num_iterations*num_messages_sent) / 2.0;
          t = (t_end - t_start) / 2.0;
        }

        else if(myrank == 1) {
          for(i = 0; i < num_iterations + num_warmup; i++) {
            for(j = 0; j < num_messages_sent; j++) {
            MPI_CHECK(MPI_Irecv(buffer_r, size, MPI_SCALAR, 0, 100, MPI_COMM_WORLD, recv_request + j));
            }

            for(j = 0; j < num_messages_sent; j++) {
            MPI_CHECK(MPI_Isend(buffer_s, size, MPI_SCALAR, 0, 10, MPI_COMM_WORLD, send_request + j));
            }

            MPI_CHECK(MPI_Waitall(num_messages_sent, send_request, reqstat));
            MPI_CHECK(MPI_Waitall(num_messages_sent, recv_request, reqstat));
          }
        }

      }

// ===================================================================================================
  else{
    if(myrank==0)
    fprintf(stderr, "Benchmark for selected routine is NOT available \n");
    MPI_CHECK(MPI_Finalize());
    exit(EXIT_FAILURE);
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (myrank == 0){
      bw = ((double) sizeof(SCALAR)*size)/(t*1024.*1024.);
      if(max_iter_time > 0){
         max_bw = ((double) sizeof(SCALAR)*size)/(min_iter_time*1024.*1024.);
         min_bw = ((double) sizeof(SCALAR)*size)/(max_iter_time*1024.*1024.);
      }

      printf(" Type-size \t Size (MB) \t Time (us) \t Avg Bandwidth (MB/s) \t Max Bandwidth (MB/s) \t Min Bandwidth (MB/s)\n");
      printf(" %3d bytes \t %9.5f \t %9.5f \t %20.3f \t %20.3f \t %20.3f\n", (int)sizeof(SCALAR), (double)nbytes/(1024*1024), t*1e6, bw, max_bw, min_bw);
    }

  sfree(buffer_s);
  sfree(buffer_r);


  MPI_CHECK(MPI_Finalize());
  return 0;
}


/* ----------------------------------------------------------------------
   safe malloc
------------------------------------------------------------------------- */
void *smalloc(int64_t nbytes)
{
    if (nbytes == 0) return NULL;
    void *ptr=NULL;

    #if defined(MEM_GPU)
    cudaMalloc((void**)&ptr, nbytes);
    xprintf("GPU memory allocation with %s \n", __func__);
    #elif defined(MEM_GPU_MAN)
    cudaMallocManaged((void**)&ptr, nbytes);
    xprintf("GPU managed memory allocation with %s \n", __func__);
    #elif defined(MEM_GPU_REG)
    ptr = malloc(nbytes);
    if (ptr) cudaHostRegister(ptr,nbytes,0);
    xprintf("GPU registered memory allocation with %s \n", __func__);
    #else
    ptr = malloc(nbytes);
    xprintf("CPU memory allocation with %s \n", __func__);
    #endif

  return ptr;
}

/* ----------------------------------------------------------------------
   safe free
------------------------------------------------------------------------- */
void sfree(void *ptr)
{
    if (ptr == NULL) return;

    #if defined(MEM_GPU)
    cudaFree(ptr);
    #elif defined(MEM_GPU_MAN)
    cudaFree(ptr);
    #elif defined(MEM_GPU_REG)
    cudaHostUnregister(ptr);
    free(ptr);
    #else
    free(ptr);
    #endif

    ptr = NULL;
}
