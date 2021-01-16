<img src="https://bitbucket.org/aayala32/logos/raw/bca97ce280291cbb051d8661990a8ba031e462f8/network.png" width="1100" height="250">


# MPI-benchmarks
*Alan Ayala - ICL-UTK*

We provide benchmark of MPI routines on CPUs and GPUs, currently we support the following types of exchange:

Point-to-Point | Collective
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
 Send  | Alltoallv 
Recv| Gatherv 
Isend | Bcast 
IRecv | Gather 
BacktoBack | Scatterv  
Bidirectional | AllGather 
- | AllReduce 

## Compilation

To exchange data between CPUs.
~~~
make mem=CPU
~~~


To exchange data between GPUs.
~~~
make mem=CPU
~~~

A helper with instruccions is displayed executing `./mpi_bw -h`.

## Running experiments

Use as follows:
~~~
<MPI_RUN_EXEC> <RESOURCES> ./mpi_bw <MPI_ROUTINE> <NUMBER_OF_ELEMENTS_TO_EXCHANGE> <NUMBER_OF_ITERATIONS>
~~~

Examples with OpenMPI:
~~~
 # Within a node:
 mpirun -n 4 ./mpi_bw  ALLGATHER    2000000      3 
 # Between 2 nodes:
 mpirun -n 2 --map-by  ./mpi_bw  SEND_BIDIR   2000000      3 
 # To get a profile: 
 mpirun -n 2 --map-by node nvprof -fo profiler_output_%q{OMPI_COMM_WORLD_RANK}.nvvp ./mpi_bw SEND 2000000  1
 ~~~

Examples with SpectrumMPI:

 ~~~
 # Within a node:
 jsrun --smpiargs="-gpu" -a1 -c1 -g1 -n2 ./mpi_bw ALL2ALLV   2000000 4 
 # Between 2 nodes:
 jsrun --smpiargs="-gpu" -r1 -a1 -c1 -g1 -n2 ./mpi_bw SEND   2000000 4 
 # To get a profile: 
 jsrun --smpiargs="-gpu" -a1 -c1 -g1 -n2 nvprof -o bw.%p.nvvp ./mpi_bw ALL2ALLV 2000000 4
 ~~~
 
 ## Experiments

 ### 1. [Summit](https://docs.olcf.ornl.gov/systems/summit_user_guide.html) @ Oak Ridge National Laboratory, USA

We show a benchmark on P2P communication using 2 ranks, 1GPU/1rank, in 2 arbitrary Summit nodes.
In the Figure below, the left part shows the interconnection peak throughput in Summit (25GB/s), on the right we show how the communication time behaves when increasing the data volume.

In practice, user can expect to get around 22.7 GB/s; i.e., 91% of theoretical peak. To estimate the exchange time for binary communication, you can use regression to derive the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=T_{P2P}&space;(\textnormal{ms})&space;=&space;\frac{M}{22.675}&space;&plus;&space;1.089," target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{P2P}&space;(\textnormal{ms})&space;=&space;\frac{M}{22.675}&space;&plus;&space;1.089," title="T_{P2P} (\textnormal{ms}) = \frac{M}{22.675} + 1.089," /></a>

where M is in MB.



<img src="https://bitbucket.org/aayala32/logos/raw/0b95cdc8ceb4f722dcf170c75a00cd1e5f31c459/summit_network.png" width="900" height="300">



  ### 2. [Fugaku](https://www.r-ccs.riken.jp/en/fugaku/project/outline) @ Riken Center, Japan

Fugaku internode connection consists in a Tofu-D + PCIe 3 external connection as showed in the figure below.


<img src="https://bitbucket.org/aayala32/logos/raw/24085cb340e15c118eeb3e10ac17364ec9a5e40e/fugaku_network.png" width="400" height="250">

