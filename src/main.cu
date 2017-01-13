#include "clock.h"
#include "parse.h"
#include "wos_kernel.cuh"

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <math_functions.h>
#include <stdio.h>

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif
//#include <cublas_v2.h>

// Source: CUDA reduction documentation
template <class T>
T reduceCPU(T *data, int size);

// this is the cumsum devided by number of relative runs (insitu)
template <typename T>
void eval2result(T *vals, int runs);

// callculate the relative error (insitu)
template <typename T>
void getRelativeError(T *vals, int runs);

template <typename T>
void outputConvergence(const char *filename, T *vals, int runs);

// initialize x0 vector of size dim and fill with val
template <typename T>
void initX0(T *x0, size_t dim, size_t len, T val);

unsigned int getRunsPerBlock(unsigned int runs, unsigned int &number_blocks);

size_t getLength(size_t dim);

int main(int argc, char *argv[]) {
  // cuda status inits
  cudaError_t cudaStat;

  // TODO differentiate between dim and len to optimally use warp size

  const size_t dim = 250;         // dimension of the problem
  size_t len = getLength(dim);    // length of the storage vector
  typedef double T;               // Type for problem
  const unsigned int runs = 1000; // number it alg itterations

  // TODO for runcount indipendent of number of blocks
  unsigned int number_blocks;
  unsigned int runsperblock = getRunsPerBlock(runs, number_blocks);

  // size variables for reduction
  Parameters p;
  parseParams(argc, argv, p);
  // int *d_runs;

  // declare local variabls
  T x0[len];
  T h_results[p.reduction.blocks];
  T h_runs[runs];
  // declare pointers for device variables
  T *d_x0;
  T *d_runs;
  T *d_results;
  // TODO: Question: what effect does the d_eps have on practical convergence?
  T d_eps = 0.01; // 1 / sqrt(dim);

  // instantiate timers
  Timer computationTime;
  Timer totalTime;

  totalTime.start();

  // init our point on host
  initX0(x0, dim, len, 0.0);

  // maloc device memory
  cudaStat = cudaMalloc((void **)&d_x0, len * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printf(" device memory allocation failed for d_x0\n");
    return EXIT_FAILURE;
  }

  cudaStat = cudaMalloc((void **)&d_runs, runs * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printf(" device memory allocation failed for d_sum\n");
    return EXIT_FAILURE;
  }

  // TODO for runcount indipendent of number of blocks
  // cudaStat = cudaMalloc((void **)&d_runs, sizeof(unsigned int));
  // if (cudaStat != cudaSuccess) {
  //   printf(" device memory allocation failed for d_sum\n");
  //   return EXIT_FAILURE;
  // }
  //
  // cudaStat = cudaMemsetAsync(d_runs, 0, sizeof(unsigned int));
  // if (cudaStat != cudaSuccess) {
  //   printf(" device memory set failed for d_runs\n");
  //   return EXIT_FAILURE;
  // }

  cudaStat = cudaMemsetAsync(d_runs, 0.0, runs * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printf(" device memory set failed for d_runs\n");
    return EXIT_FAILURE;
  }

  // Let's bing our data to the Device
  cudaStat = cudaMemcpyAsync(d_x0, x0, len * sizeof(T), cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess) {
    printf(" device memory upload failed\n");
    return EXIT_FAILURE;
  }
  // TODO make dim power of 2 and min 1 warp for reductions

  computationTime.start();

  // Calling WoS kernel
  wos<T>(number_blocks, len, d_x0, d_runs, d_eps, dim, runsperblock,
         getSizeSharedMem<T>(len));

  cudaDeviceSynchronize();
  computationTime.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

  cudaStat = cudaMemcpyAsync(&h_runs, d_runs, runs * sizeof(T),
                             cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }

  // 256 blocks in global reduce
  cudaStat = cudaMalloc((void **)&d_results, p.reduction.blocks * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printf(" device memory allocation failed for d_sum\n");
    return EXIT_FAILURE;
  }

// convergence plot export

#ifdef PLOT
  printf("exporting convergences data\n");
  eval2result(h_runs, runs);
  getRelativeError(h_runs, runs);
  outputConvergence("docs/data/cuWos_convergence.dat", h_runs, runs);
// outputRuntime();
#endif

  // perform local reducion on CPU
  reduce(runs, p.reduction.threads, p.reduction.blocks, d_runs, d_results);

  cudaStat = cudaMemcpy(&h_results, d_results, p.reduction.blocks * sizeof(T),
                        cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }
  h_results[0] = reduceCPU(h_results, p.reduction.blocks);

  totalTime.end();

  printf("average: %f \nrunning time: %f sec  \ntotal time: %f sec \n",
         h_results[0] / runs, computationTime.get(), totalTime.get());
  cudaFree(d_results);

  return (0);
}

size_t getLength(size_t dim) {
  size_t len;
  if (isPow2(dim)) {
    printf("dimension is power of 2\n");
    len = dim;
  } else {
    printf("dimensions length expanded to next pow2\n");
    len = nextPow2(dim);
  }
#ifdef DEBUG
  printf("value of len is: \t%lu \n", len);
  printf("value of dim is: \t%lu \n", dim);
#endif
  return len;
}

unsigned int getRunsPerBlock(unsigned int runs, unsigned int &number_blocks) {
  unsigned int runsperblock = 1;
  number_blocks = runs;
  if (runs > MAX_BLOCKS) {
    while (MAX_BLOCKS < number_blocks) {
      runsperblock++;
      number_blocks /= runsperblock;
    }
    printf("runs: %d\nnumber of blocks: %d \n runs per block: %d\n", runs,
           number_blocks, runsperblock);
  }
  return runsperblock;
}

// this is the cumsum devided by number of relative runs (insitu)
template <typename T>
void eval2result(T *vals, int runs) {
  for (int i = 1; i < runs; i++) {
    vals[i] += vals[i - 1];
    vals[i - 1] /= i;
  }
  vals[runs - 1] /= runs;
}
// callculate the relative error (insitu)
template <typename T>
void getRelativeError(T *vals, int runs) {
  T end = vals[runs - 1];
  for (int i = 0; i < runs; i++) {
    vals[i] = abs((vals[i] - end) / end);
  }
}
template <typename T>
void outputConvergence(const char *filename, T *vals, int runs) {
  // BUG
  // TODO impliment for run numbers greater than MAX_BLOCKS
  std::ofstream file(filename);
  file << "run\t"
       << "solution val\t" << std::endl;
  // only export every 10th val reduce file size
  for (int i = 0; i < runs; i += 10) {
    file << i << "\t" << vals[i] / i << "\t" << std::endl;
  }
  file.close();
}
template <typename T>
void initX0(T *x0, size_t dim, size_t len, T val) {
  // init our point on host
  for (unsigned int i = 0; i < dim; i++)
    // x0[i] = i == 1 ? 0.22 : 0;
    x0[i] = val;
  for (unsigned int i = dim; i < len; i++)
    x0[i] = 0.0;
}
