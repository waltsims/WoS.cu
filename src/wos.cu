#include "clock.h"
#include "wos_kernel.cuh"

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <math_functions.h>
#include <stdio.h>

//#include <cublas_v2.h>

// Source: CUDA reduction documentation
////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T> T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}
// this is the cumsum devided by number of relative runs (insitu)
template <typename T> void eval2result(T *vals, int runs) {
  for (int i = 1; i < runs; i++) {
    vals[i] += vals[i - 1];
    vals[i - 1] /= i;
  }
  vals[runs - 1] /= runs;
}
// callculate the relative error (insitu)
template <typename T> void getRelativeError(T *vals, int runs) {
  T end = vals[runs - 1];
  for (int i = 0; i < runs; i++) {
    vals[i] = abs((vals[i] - end) / end);
  }
}
template <typename T>
void outputConvergence(const char *filename, T *vals, int runs) {

  std::ofstream file(filename);
  file << "run\t"
       << "solution val\t" << std::endl;
  // only export every 10th val reduce file size
  for (int i = 0; i < runs; i += 10) {
    file << i << "\t" << vals[i] / i << "\t" << std::endl;
  }
  file.close();
}

int main(int argc, char *argv[]) {
  // cuda status inits
  cudaError_t cudaStat;

  // TODO differentiate between dim and len to optimally use warp size

  const size_t dim = 3; // dimension of the problem
  size_t len;           // length of the storage vector

  if (isPow2(dim)) {
    printf("dimension is power of 2\n");
    len = dim;
  } else {
    printf("dimensions length should be expanded to next pow2\n");
    len = nextPow2(dim);
  }
  printf("value of len is: \t%lu \n", len);
  printf("value of dim is: \t%lu \n", dim);

  int blocks = 256;
  int threads = 512;
  typedef double T;
  const unsigned int runs = 65535;
  // TODO for runcount indipendent of number of blocks
  // int *d_runs;

  T x0[len];
  T h_results[blocks];
  T h_runs[runs];
  T *d_x0;
  T *d_runs;
  T *d_results;
  // TODO: what effect does the d_eps have on practical convergence?
  T d_eps = 0.01; // 1 / sqrt(dim);
  Timer computationTime;
  Timer totalTime;

  totalTime.start();
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

  // 256 bloks in global reduce
  cudaStat = cudaMalloc((void **)&d_results, blocks * sizeof(T));
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

  // init our point on host
  for (int i = 0; i < dim; i++)
    // x0[i] = i == 1 ? 0.22 : 0;
    x0[i] = 0.0;
  for (int i = dim; i < len; i++)
    x0[i] = 0.0;

  // Let's bing our data to the Device
  cudaStat = cudaMemcpyAsync(d_x0, x0, len * sizeof(T), cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess) {
    printf(" device memory upload failed\n");
    return EXIT_FAILURE;
  }
  // TODO make dim power of 2 and min 1 warp for reductions

  // Calling WoS kernel
  computationTime.start();
  // TODO cudastat return value for kernel calls
  WoS<T><<<runs, len, (4 * len + 1) * sizeof(T)>>>(d_x0, d_runs, d_eps, dim,
                                                   len);
  cudaDeviceSynchronize();
  computationTime.end();

  // convergence plot export
  cudaStat =
      cudaMemcpy(&h_runs, d_runs, runs * sizeof(T), cudaMemcpyDeviceToHost);
#ifdef PLOT
  printf("exporting convergences data\n");
  eval2result(h_runs, runs);
  getRelativeError(h_runs, runs);
  outputConvergence("docs/data/cuWos_convergence.dat", h_runs, runs);
#endif

  reduce(runs, threads, blocks, d_runs, d_results);

  cudaStat = cudaMemcpy(&h_results, d_results, blocks * sizeof(T),
                        cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }
  h_results[0] = reduceCPU(h_results, blocks);

  totalTime.end();

  printf("average: %f  \nrunning time: %f sec  \ntotal time: %f sec \n",
         h_results[0] / runs, computationTime.get(), totalTime.get());
  cudaFree(d_results);
  cudaFree(d_x0);

  return (0);
}
