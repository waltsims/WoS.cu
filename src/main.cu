#include "clock.h"
#include "parse.h"
#include "wos_kernel.cuh"

#include <fstream>
#include <limits>
#include <math_functions.h>

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif
//#include <cublas_v2.h>

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

int main(int argc, char *argv[]) {
  printTitle();
  // cuda status inits
  printInfo("initializing");
  cudaError_t cudaStat;
  Parameters p;

  // TODO this should/could go in parameter constructor
  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

  // TODO: call WoS template wraper function
  // if (p.wos.typeDouble) {
  typedef double T; // Type for problem

  // } else {
  //   typedef float T; // Type for problem
  // }

  // TODO: Question: what effect does the d_eps have on practical convergence?
  T d_eps = 0.01; // 1 / sqrt(p.wos.x0.dimension); // or 0.01

  // instantiate timers
  Timer computationTime;
  Timer totalTime;
  Timer memoryTime;

  totalTime.start();

  // declare local array variabls
  T x0[p.wos.x0.length];
  // declare pointers for device variables
  T *d_x0;
  T *d_runs;
  // init our point on host
  // cast to T hotfix until class is templated
  initX0(x0, p.wos.x0.dimension, p.wos.x0.length, (T)p.wos.x0.value);

  memoryTime.start();

  // maloc device memory
  cudaStat = cudaMalloc((void **)&d_x0, p.wos.x0.length * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory allocation failed for d_x0");
    return EXIT_FAILURE;
  }

  printf("initializing d_runs with a length of %d\n", p.wos.totalPaths);

  cudaStat = cudaMalloc((void **)&d_runs, p.wos.totalPaths * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory allocation failed for d_sum");
    return EXIT_FAILURE;
  }

  // TODO for runcount independant of number of blocks

  cudaStat = cudaMemset(d_runs, 0.0, p.wos.totalPaths * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory set failed for d_runs");
    return EXIT_FAILURE;
  }

  // Let's bring our data to the Device
  cudaStat =
      cudaMemcpy(d_x0, x0, p.wos.x0.length * sizeof(T), cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess) {
    printError("device memory upload failed");
    return EXIT_FAILURE;
  }

  float prep = memoryTime.get();
  computationTime.start();

  // Calling WoS kernel
  // TODO pass only struct to wos
  wos<T>(p, d_x0, d_runs, d_eps);

  cudaDeviceSynchronize();
  computationTime.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

#ifdef PLOT
  // create variable for Plot data
  T h_runs[p.wos.totalPaths];
  // convergence plot export
  cudaStat = cudaMemcpyAsync(&h_runs, d_runs, p.wos.totalPaths * sizeof(T),
                             cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }

  printf("exporting convergences data\n");
  eval2result(h_runs, p.wos.totalPaths);
  getRelativeError(h_runs, p.wos.totalPaths);
  outputConvergence("docs/data/cuWos_convergence.dat", h_runs,
                    p.wos.totalPaths);
// outputRuntime();
#endif
  memoryTime.start();

#ifdef CPU_REDUCE
  T h_runs[p.wos.totalPaths];
  // convergence plot export
  cudaStat = cudaMemcpyAsync(&h_runs, d_runs, p.wos.totalPaths * sizeof(T),
                             cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }

  float mid = memoryTime.get() - prep;

  T gpu_result = reduceCPU(h_runs, p.wos.totalPaths);

#else
  // T h_results[p.reduction.blocks];

  T *h_results = (T *)malloc(p.reduction.blocks * sizeof(T));
  // init h_results:
  for (int j = 0; j < p.reduction.blocks; j++) {
    h_results[j] = 0.0;
  }

  T *d_results;
  cudaStat = cudaMalloc((void **)&d_results, p.reduction.blocks * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory allocation failed for d_results");
    return EXIT_FAILURE;
  }

  float mid = memoryTime.get() - prep;

  // if (p.reduction.blocks > 1) {
  cudaError err;
  reduce(p.wos.totalPaths, p.reduction.threads, p.reduction.blocks, d_runs,
         d_results);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Reduction Kernel returned an error:\n %s\n",
           cudaGetErrorString(err));
  }
  //  }

  memoryTime.start();

#ifdef DEBUG
  printf("[MAIN]: results values before copy:\n");
  for (int n = 0; n < p.reduction.blocks; n++) {
    printf("%f\n", h_results[n]);
  }
#endif

  // copy result from device to hostcudaStat =
  cudaStat = cudaMemcpy(h_results, d_results, p.reduction.blocks * sizeof(T),
                        cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printError("device memory download failed");
    return EXIT_FAILURE;
  }

  T gpu_result = 0.0;
  for (int i = 0; i < p.reduction.blocks; i++) {
    printf("iteration %d, %f\n", i, h_results[i]);
    gpu_result += h_results[i];
  }
  free(h_results);
#endif
  gpu_result /= p.wos.totalPaths;

#ifdef DEBUG
  printf("[MAIN]: results values after copy:\n");
  for (int n = 0; n < p.reduction.blocks; n++) {
    printf("%f\n", h_results[n]);
  }
#endif

  float finish = memoryTime.get() - mid - prep;

  totalTime.end();

  testResults((float)x0[0], (float)d_eps, (float)gpu_result, p);

  printTiming(prep, computationTime.get(), totalTime.get(), finish);

#ifndef CPU_REDUCE
  cudaFree(d_results);
#endif

  return (0);
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
