#ifndef THRUST

#include <curand_kernel.h>
#include <iostream>

#include "params.h"
#include "reduce_kernel.cuh"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

template <typename T>
__device__ void warpMinReduce(T *sdata, int tid);

template <typename T>
__device__ void warpSumReduce(T *sdata, int tid);

template <typename T>
__device__ void broadcast(T *sdata, int tid);

template <typename T>
__device__ void project2Boundary(T *s_x, T *cache, size_t dim, int tid);

template <typename T>
__device__ void norm2(T *s_radius, int tid);

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid);

template <typename T>
__device__ void getBoundaryDistance(T *s_cache, T *d_x, int tid);

template <typename T>
__device__ void evaluateBoundaryValue(T *s_x, T *s_cache, T *d_result,
                                      const size_t dim, int tid);

template <typename T>
struct BlockVariablePointers {
  T *s_radius, *s_direction, *s_cache, *s_x;
};

template <typename T>
__device__ void calcSubPointers(BlockVariablePointers<T> *bvp, size_t len,
                                T *buff) {
  bvp->s_radius = buff;
  bvp->s_direction = len + buff;
  bvp->s_cache = 2 * len + buff;
  bvp->s_x = 3 * len + buff;
}
template <typename T>
__device__ BlockVariablePointers<T> smemInit(BlockVariablePointers<T> bvp,
                                             T *d_x0, int tid) {
  // initialize shared memory
  bvp.s_direction[tid] = 0.0;
  bvp.s_cache[tid] = 0.0;
  bvp.s_radius[tid] = INFINITY;
  // copy x0 to local __shared__"moveable" x
  bvp.s_x[tid] = d_x0[tid];
  return bvp;
}

template <typename T>
__global__ void WoS(T *d_x0, T *d_global, T d_eps, size_t dim, size_t len,
                    unsigned int pathsPerBlock) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ T buff[];
  BlockVariablePointers<T> bvp;
  calcSubPointers(&bvp, len, buff);
  smemInit(bvp, d_x0, tid);

#ifdef DEBUG
  if (tid == 0)
    printf("[WOS]: d_global[%d] before:\t%f\n", blockIdx.x,
           d_global[blockIdx.x]);
  __syncthreads();
#endif

  curandState s;
  // seed for random number generation
  unsigned int seed = index;

  // TODO: x0 in texture meomry
  curand_init(seed, 0, 0, &s);
  T r = INFINITY;
  // max step size
  while (d_eps < r) {

    getBoundaryDistance<T>(bvp.s_radius, bvp.s_x, tid);

    warpMinReduce<T>(bvp.s_radius, tid);
    // local copy of radius
    r = bvp.s_radius[0];

    // random next step_direction
    bvp.s_direction[tid] = (tid < dim) * curand_normal(&s);

    // normalize direction with L2 norm
    normalize<T>(bvp.s_direction, bvp.s_cache, dim, tid);

    // next x point
    bvp.s_x[tid] += r * bvp.s_direction[tid];
  }

  // find closest boundary point
  project2Boundary<T>(bvp.s_x, bvp.s_cache, dim, tid);

  // boundary eval and return do global memory
  evaluateBoundaryValue<T>(bvp.s_x, bvp.s_cache, d_global, dim, tid);
#ifdef DEBUG
  if (tid == 0) {
    printf("[WOS]: result on block %d:\n", blockIdx.x);
    printf("%f, ", d_global[blockIdx.x]);
    printf("\n");
  }
  __syncthreads();
#endif
}

template <typename T>
__device__ void warpMinReduce(T *sdata, int tid) {
  // each thread puts its local value into warp variable
  T myMin = sdata[tid];
  unsigned int blockSize = blockDim.x;

  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 256])) ? myMin : sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 128])) ? myMin : sdata[tid + 128];
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 64])) ? myMin : sdata[tid + 64];
  }

  __syncthreads();

#if (__CUDA_ARCH__ >= 300)
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      myMin = ((abs(myMin)) < abs(sdata[tid + 32])) ? myMin : sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      T temp = __shfl_down(myMin, offset);
      myMin = (abs(temp) < abs(myMin)) ? temp : myMin;
    }
  }
#else
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) {

    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 32])) ? myMin : sdata[tid + 32];
  }

  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 16])) ? myMin : sdata[tid + 16];
  }

  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 8])) ? myMin : sdata[tid + 8];
  }

  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 4])) ? myMin : sdata[tid + 4];
  }

  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 2])) ? myMin : sdata[tid + 2];
  }

  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 1])) ? myMin : sdata[tid + 1];
  }

  __syncthreads();
#endif

  if (tid == 0)
    sdata[0] = myMin;

  __syncthreads();
}

template <typename T>
__device__ void warpSumReduce(T *sdata, int tid) {
  // each thread puts its local sum value into warp variable
  T mySum = sdata[tid];
  unsigned int blockSize = blockDim.x;

  // do reduction in shared mem

  if ((blockSize >= 1024) && (tid < 512)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  __syncthreads();

#if (__CUDA_ARCH__ >= 300)
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      mySum += __shfl_down(mySum, offset);
    }
  }
#else
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) {
    sdata[tid] = mySum = mySum + sdata[tid + 32];
  }

  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata[tid] = mySum = mySum + sdata[tid + 16];
  }

  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata[tid] = mySum = mySum + sdata[tid + 8];
  }

  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata[tid] = mySum = mySum + sdata[tid + 4];
  }

  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata[tid] = mySum = mySum + sdata[tid + 2];
  }

  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata[tid] = mySum = mySum + sdata[tid + 1];
  }

  __syncthreads();
#endif

  if (tid == 0)
    sdata[0] = mySum;

  __syncthreads();
}

template <typename T>
__device__ void broadcast(T *sdata, int tid) {

  if (tid != 0) // needed for race condition check
    sdata[tid] = sdata[0];
  __syncthreads(); // needed for race condition check
}

template <typename T>
__device__ void project2Boundary(T *s_x, T *cache, size_t dim, int tid) {
  getBoundaryDistance(cache, s_x, tid);
  warpMinReduce(cache, tid);
  broadcast(cache, tid);

  // TODO: could be faster if index of min was known
  s_x[tid] = ((1.0 - abs(s_x[tid])) == cache[tid]) ? 1.0 : s_x[tid];
  __syncthreads();
}

template <typename T>
__device__ void norm2(T *s_radius, T *s_cache, int tid) {

  // square vals and copy to cache
  s_cache[tid] = s_radius[tid] * s_radius[tid];

  warpSumReduce<T>(s_cache, tid);

  if (tid == 0)
    s_cache[tid] = sqrt(s_cache[tid]);
  __syncthreads();
}

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid) {

  norm2(s_radius, cache, tid);
  s_radius[tid] /= cache[0];
  //__syncthreads(); // needed for race check
}

template <typename T>
__device__ void getBoundaryDistance(T *s_cache, T *d_x, int tid) {
  s_cache[tid] = 1.0 - abs(d_x[tid]);
}

template <typename T>
__device__ void evaluateBoundaryValue(T *s_x, T *s_cache, T *d_result,
                                      const size_t dim, int tid) {

  s_cache[tid] = s_x[tid] * s_x[tid];

  warpSumReduce<T>(s_cache, tid);

  if (tid == 0) {
#ifdef DEBUG
    printf("[WOS]: output from block %d:\t%f\n", blockIdx.x,
           s_cache[0] / (2 * dim));
#endif
    d_result[blockIdx.x] += s_cache[0] / (2 * dim);
  }
  __syncthreads();
}

//==============================================================================
template <typename T>
void initX0(T *x0, size_t dim, size_t len, T val);

template <typename T>
T wosNative(Timers &timers, Parameters &p) {

  // declare local array variabls
  T *h_x0;
  checkCudaErrors(cudaMallocHost((void **)&h_x0, sizeof(T) * p.wos.x0.length));
  // declare pointers for device variables
  T *d_x0 = NULL;
  T *d_paths = NULL;
  // init our point on host
  // cast to T hotfix until class is templated
  T d_eps = p.wos.eps;
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
  cudaFreeHost(h_x0);

  timers.memorySetupTimer.end();
  timers.computationTimer.start();

  printInfo("setting up problem");
  dim3 dimBlock(p.wos.x0.length, 1, 1);
  dim3 dimGrid(p.wos.totalPaths, 1, 1);

  cudaError err;

  WoS<T><<<dimGrid, dimBlock, p.wos.size_SharedMemory>>>(
      d_x0, d_paths, d_eps, p.wos.x0.dimension, p.wos.x0.length,
      p.wos.pathsPerBlock);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Wos Kernel returned an error:\n %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  timers.computationTimer.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

  printInfo("downloading path data");
  timers.memoryDownloadTimer.start();
  T h_paths[p.wos.totalPaths];
  // Download paths data
  checkCudaErrors(cudaMemcpyAsync(
      &h_paths, d_paths, p.wos.totalPaths * sizeof(T), cudaMemcpyDeviceToHost));

  timers.memoryDownloadTimer.end();

#ifdef OUT
  exportData(h_paths, p);
#endif // OUT

  printInfo("reduce data on CPU");
  T gpu_result = reduceCPU(h_paths, p.wos.totalPaths);

  gpu_result /= p.wos.totalPaths;

#ifdef DEBUG
  printf("[MAIN]: results values after copy:\n");
  for (unsigned int n = 0; n < p.wos.totalPaths; n++) {
    printf("%f\n", h_paths[n]);
  }
#endif

  timers.totalTimer.end();

  cudaFree(d_paths);
  return gpu_result;
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
#endif // !THRUST
