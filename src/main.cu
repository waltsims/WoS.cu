#include "clock.h"
#include "helper.hpp"
#include "parse.h"
#include "wos_kernel.cuh"

#include <fstream>
#include <limits>
#include <math.h>
#include <math_functions.h>

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

void printTitle();
void printInfo(const char *info);

unsigned int getRunsPerBlock(unsigned int runs, unsigned int &number_blocks);

size_t getLength(size_t dim);

int main(int argc, char *argv[]) {
  printTitle();
  // cuda status inits
  printInfo("initializing");
  cudaError_t cudaStat;
  Parameters p;

  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

  p.wos.x0.length =
      getLength(p.wos.x0.dimension); // length of the storage vector
  typedef double T;                  // Type for problem

  // TODO for runcount indipendent of number of blocks
  unsigned int number_blocks;
  unsigned int runsperblock = getRunsPerBlock(p.wos.iterations, number_blocks);

  // declare local array variabls
  T x0[p.wos.x0.length];
  T h_results[p.reduction.blocks];
  // declare pointers for device variables
  T *d_x0;
  T *d_runs;
  T *d_results;
  // TODO: Question: what effect does the d_eps have on practical convergence?
  T d_eps = 0.01; // 1 / sqrt(p.wos.x0.dimension); // or 0.01

  // instantiate timers
  Timer computationTime;
  Timer totalTime;
  Timer memoryTime;

  totalTime.start();

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

  cudaStat = cudaMalloc((void **)&d_runs, p.wos.iterations * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory allocation failed for d_sum");
    return EXIT_FAILURE;
  }

  // TODO for runcount independant of number of blocks

  cudaStat = cudaMemsetAsync(d_runs, 0.0, p.wos.iterations * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory set failed for d_runs");
    return EXIT_FAILURE;
  }

  // Let's bring our data to the Device
  cudaStat = cudaMemcpyAsync(d_x0, x0, p.wos.x0.length * sizeof(T),
                             cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess) {
    printError("device memory upload failed");
    return EXIT_FAILURE;
  }

  float prep = memoryTime.get();
  computationTime.start();

  // Calling WoS kernel
  wos<T>(number_blocks, p.wos.x0.length, d_x0, d_runs, d_eps,
         p.wos.x0.dimension, runsperblock,
         getSizeSharedMem<T>(p.wos.x0.length));

  cudaDeviceSynchronize();
  computationTime.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

#ifdef PLOT
  // create variable for Plot data
  T h_runs[p.wos.itterations];
  // convergence plot export
  cudaStat = cudaMemcpyAsync(&h_runs, d_runs, p.wos.itterations * sizeof(T),
                             cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }

  printf("exporting convergences data\n");
  eval2result(h_runs, runs);
  getRelativeError(h_runs, runs);
  outputConvergence("docs/data/cuWos_convergence.dat", h_runs, runs);
// outputRuntime();
#endif
  memoryTime.start();

  cudaStat = cudaMalloc((void **)&d_results, p.reduction.blocks * sizeof(T));
  if (cudaStat != cudaSuccess) {
    printError("device memory allocation failed for d_sum");
    return EXIT_FAILURE;
  }

  float mid = memoryTime.get() - prep;
  // perform local reducion on CPU
  reduce(p.wos.iterations, p.reduction.threads, p.reduction.blocks, d_runs,
         d_results);

  memoryTime.start();

  cudaStat =
      cudaMemcpyAsync(&h_results, d_results, p.reduction.blocks * sizeof(T),
                      cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printError("device memory download failed");
    return EXIT_FAILURE;
  }
  h_results[0] = reduceCPU(h_results, p.reduction.blocks);

  T result = h_results[0] / p.wos.iterations;

  float finish = memoryTime.get() - mid - prep;

  totalTime.end();

  // TODO: dynamic table output function
  // this is a hot mess

  // Basic testing

  printf("\nSIMULATION SUMMARY: \n\n");
  printf("VALUES: \n");

  printf(" ----------------------------------------------------------------"
         "------------------------------\n");
  printf("|%-15s|%-15s|%-15s|%-15s|%-15s|%-15s|\n", "iterations",
         "desired value", "resulting value", "epsilon", "delta", "status");
  printf(" ----------------------------------------------------------------"
         "------------------------------\n");

  T EPS = 0.00001;
  T desired = 0.0;

  if (abs(x0[0] - 0.0) < EPS) {
    desired = (d_eps != 0.01) * 0.039760;
    desired = (d_eps == 0.01) * 0.042535;
    // Julia value [0.0415682]
    if (abs(result - desired) < EPS) {
      printf("|%-15d|%-15lf|%-15f|%-15f|%-15f|", p.wos.iterations, desired,
             result, EPS, abs(result - desired));
      printf(ANSI_GREEN "%-14s" ANSI_RESET, "TEST PASSED!");
      printf("|\n");
    } else {
      printf("|%-15d|%-15lf|%-15f|%-15f|%-15f|", p.wos.iterations, desired,
             result, EPS, abs(result - desired));
      printf(ANSI_RED "%-14s" ANSI_RESET, "TEST FAILED!");
      printf("|\n");
    }
  } else if (abs(x0[0] - 1.0) < EPS) {
    T desired = 0.5;
    if (abs(result - desired) < EPS) {
      printf("|%-15d|%-15lf|%-15f|%-15f|%-15f|", p.wos.iterations, desired,
             result, EPS, abs(result - desired));
      printf(ANSI_GREEN "%-14s" ANSI_RESET, "TEST PASSED!");
      printf("|\n");
    } else {
      printf("|%-15d|%-15lf|%-15f|%-15f|%-15f|", p.wos.iterations, desired,
             result, EPS, abs(result - desired));
      printf(ANSI_RED "%-14s" ANSI_RESET, "TEST FAILED!");
      printf("|\n");
    }
  }
  printf(" ----------------------------------------------------------------"
         "------------------------------\n");
  // Time output
  printf("TIMING: \n");

  printf(" ----------------------------------------------------------------"
         "----------------------------------------\n");
  printf("|%-25s|%-25s|%-25s|%-25s|\n", "memory init time[sec]",
         "GPU computation time[sec]", "total exicution time[sec]",
         "memory finish time[sec]");
  printf(" ----------------------------------------------------------------"
         "----------------------------------------\n");
  printf("|%-25f|%-25f|%-25f|%-25f|\n ", prep, computationTime.get(),
         totalTime.get(), finish);
  printf(" ----------------------------------------------------------------"
         "---------------------------------------\n");
  cudaFree(d_results);

  return (0);
}

size_t getLength(size_t dim) { return (isPow2(dim)) ? dim : nextPow2(dim); }

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
