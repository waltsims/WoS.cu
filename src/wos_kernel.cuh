#include <curand_kernel.h>
#include <iostream>

#include "params.h"
#include "reduce_kernel.cuh"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid);

template <typename T>
__device__ void warpMinReduce(T *sdata, int tid);

template <typename T>
__device__ void warpReduce(T *sdata, int tid);

template <typename T>
__device__ void broadcast(T *sdata, int tid);

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid);

template <typename T>
__device__ void WoS(T *s_cache, int tid);

template <typename T>
__device__ void norm2(T *s_radius, int tid);

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid);

template <typename T>
__device__ void boundaryDistance(T *s_cache, T *d_x, int tid);

template <typename T>
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *d_result,
                                 const size_t dim, int tid, int blockRun);

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
                    int runsperblock) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ T buff[];
  BlockVariablePointers<T> bvp;
  calcSubPointers(&bvp, len, buff);
  smemInit(bvp, d_x0, tid);

  for (int i = 0; i < runsperblock; i++) {
    // seed for random number generation
    unsigned int seed = index;
    curandState s;
    curand_init(seed, 0, 0, &s);

    // TODO: x0 in texture meomry

    T r = INFINITY;
    // max step size
    while (d_eps < r) {

      boundaryDistance<T>(bvp.s_radius, bvp.s_x, tid);

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
    round2Boundary<T>(bvp.s_x, bvp.s_cache, dim, tid);

    // boundary eval and return do global memory
    evaluateBoundary<T>(bvp.s_x, bvp.s_cache, d_global, dim, tid, i);
  }
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
__device__ void warpReduce(T *sdata, int tid) {
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

  // if (tid != 0) // needed for race condition check
  sdata[tid] = sdata[0];
  //__syncthreads(); // needed for race condition check
}

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid) {
  boundaryDistance(cache, s_x, tid);
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

  warpReduce<T>(s_cache, tid);

  if (tid == 0) {
    s_cache[tid] = sqrt(s_cache[tid]);
#ifdef DEBUG
    printf("the 2norm of the value r is %f\n", s_cache[0]);
#endif
  }
  __syncthreads();
}

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid) {

  norm2(s_radius, cache, tid);
  s_radius[tid] /= cache[0];
//__syncthreads(); // needed for race check
#ifdef DEBUG
  printf("normalized value on thread %d after normilization: %f \n", tid,
         s_radius[tid]);
#endif
}

template <typename T>
__device__ void boundaryDistance(T *s_cache, T *d_x, int tid) {
  s_cache[tid] = 1.0 - abs(d_x[tid]);
}

template <typename T>
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *d_result,
                                 const size_t dim, int tid, int blockRun) {

  s_cache[tid] = s_x[tid] * s_x[tid];

  warpReduce<T>(s_cache, tid);

  if (tid == 0) {
    d_result[blockIdx.x + blockDim.x * blockRun] = s_cache[0] / (2 * dim);
  }
  __syncthreads();
}

//==============================================================================
template <typename T>
void wos(unsigned int blocks, size_t threads, T *d_x0, T *d_runs, T d_eps,
         const int dim, unsigned int runsperblock, size_t smemSize) {

  printInfo("setting up problem");
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  cudaError err;

  WoS<T><<<dimGrid, dimBlock, smemSize>>>(d_x0, d_runs, d_eps, dim, threads,
                                          runsperblock);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Wos Kernel returned an error:\n %s\n", cudaGetErrorString(err));
  }
}
