#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include "inc/helper_cuda.h"
#include "src/clock.h"
#include "src/export.h"
#include "src/parse.h"

#include "src/wos_kernel.cuh"
#include "src/wos_thrust.h"

#include <limits>
#include <math_functions.h>

// initialize h_x0 vector of size dim and fill with val
template <typename T>
void initX0(T *x0, size_t dim, size_t len, T val);

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

// TODO: call WoS template wraper function

#ifdef DOUBLE
  typedef double T; // Type for problem
#else
  typedef float T;
#endif // DOUBLE
  Parameters p;

  // TODO this should/could go in parameter constructor
  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

  // instantiate timers
  Timers timers;

  timers.totalTimer.start();
#ifndef THRUST
  // declare local array variabls
  T *h_x0;
  checkCudaErrors(cudaMallocHost((void **)&h_x0, sizeof(T) * p.wos.x0.length));
  // declare pointers for device variables
  T *d_x0 = NULL;
  T *d_paths = NULL;
  // init our point on host
  // cast to T hotfix until class is templated
  initX0(h_x0, p.wos.x0.dimension, p.wos.x0.length, (T)p.wos.x0.value);

  timers.memorySetupTimer.start();

  // maloc device memory
  checkCudaErrors(cudaMalloc((void **)&d_x0, p.wos.x0.length * sizeof(T)));

  printInfo("initializing d_paths");

  checkCudaErrors(cudaMalloc((void **)&d_paths, p.wos.totalPaths * sizeof(T)));

  checkCudaErrors(cudaMemset(d_paths, 0.0, p.wos.totalPaths * sizeof(T)));

  // Let's bring our data to the Device
  checkCudaErrors(cudaMemcpy(d_x0, h_x0, p.wos.x0.length * sizeof(T),
                             cudaMemcpyHostToDevice));

  timers.memorySetupTimer.end();
  timers.computationTimer.start();

  // Calling WoS kernel
  wos<T>(p, d_x0, d_paths, d_eps);
  cudaDeviceSynchronize();
  timers.computationTimer.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

#else  // THRUST
  // T gpu_result = wosThrust<T>(timers, p);

  double gpu_result = wosThrust(timers, p);
#endif // NTHRUST
  timers.totalTimer.end();

#ifndef THRUST
#if defined(OUT) || defined(CPU_REDUCE)
  printInfo("downloading path data");
  timers.memoryDownloadTimer.start();

  T h_paths[p.wos.totalPaths];
  // Download paths data
  checkCudaErrors(cudaMemcpyAsync(
      &h_paths, d_paths, p.wos.totalPaths * sizeof(T), cudaMemcpyDeviceToHost));

  timers.memoryDownloadTimer.end();

  exportData(h_paths, p);
#endif // OUT || CPU_REDUCE

#ifdef CPU_REDUCE

  printInfo("reduce data on CPU");
  T gpu_result = reduceCPU(h_paths, p.wos.totalPaths);

#else //GPU_REDCUE

  T *h_results = (T *)malloc(p.reduction.blocks * sizeof(T));

  T *d_results;
  checkCudaErrors(
      cudaMalloc((void **)&d_results, p.reduction.blocks * sizeof(T)));

  cudaError err;
  reduce(p.wos.totalPaths, p.reduction.threads, p.reduction.blocks, d_paths,
         d_results);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Reduction Kernel returned an error:\n %s\n",
           cudaGetErrorString(err));
  }

  timers.memoryDownloadTimer.start();

  printf("[MAIN]: results values before copy:\n");
  for (int n = 0; n < p.reduction.blocks; n++) {
    printf("%f\n", h_results[n]);
  }
#endif //CPU_REDUCE

  // copy result from device to hostcudaStat =
  checkCudaErrors(cudaMemcpy(h_results, d_results,
                             p.reduction.blocks * sizeof(T),
                             cudaMemcpyDeviceToHost));

  timers.memoryDownloadTimer.end();

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

  timers.totalTimer.end();


#ifndef THRUST
  testResults((float)d_eps, (float)gpu_result, p);
  cudaFreeHost(h_x0);
#endif // !THRUST

  printTiming(timers.memorySetupTimer.get(), timers.computationTimer.get(),
              timers.totalTimer.get(), timers.memoryDownloadTimer.get());

#ifndef CPU_REDUCE
  cudaFree(d_results);
#endif

  return (0);
}

template <typename T>
void initX0(T *h_x0, size_t dim, size_t len, T val) {
  // init our point on host
  for (unsigned int i = 0; i < dim; i++)
    // h_x0[i] = i == 1 ? 0.22 : 0;
    h_x0[i] = val;
  for (unsigned int i = dim; i < len; i++)
    h_x0[i] = 0.0;
}
