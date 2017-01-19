#include <curand_kernel.h>
#include <iostream>

#include "params.h"
#include "reduce_kernel.cuh"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid);

template <typename T>
__device__ void warpReduce(T *sdata, int tid);

template <typename T>
__device__ void broadcast(T *s_radius, int tid);

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
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *s_result,
                                 const size_t dim, int tid);

// BUG: pointer structer is currently in local memory and ruining performance
template <typename T>
struct BlockVariablePointers {
  T *s_radius, *s_direction, *s_cache, *s_x, *s_result;
};
template <typename T>
__device__ void calcSubPointers(BlockVariablePointers<T> *bvp, size_t len,
                                T *buff) {
  bvp->s_radius = buff;
  bvp->s_direction = len + buff;
  bvp->s_cache = 2 * len + buff;
  bvp->s_x = 3 * len + buff;
  bvp->s_result = 4 * len + buff;
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
  if (tid == 0)
    bvp.s_result[0] = 0.0;
  __syncthreads();
  return bvp;
}

template <typename T>
__global__ void WoS(T *d_x0, T *d_global, T d_eps, size_t dim, size_t len,
                    int runsperblock) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // shared buff should currently be 4 * problem size + 1
  extern __shared__ T buff[];
  // bvp WARP variable
  // is this global or Register var?
  BlockVariablePointers<T> bvp;
  calcSubPointers(&bvp, len, buff);
  smemInit(bvp, d_x0, tid);

  for (int i = 0; i < runsperblock; i++) {
    // seed for random number generation
    unsigned int seed = index;
    curandState s;
    curand_init(seed, 0, 0, &s);

    // TODO: x0 in constmem

    T r = INFINITY;
    // max step size
    while (d_eps < r) {

      boundaryDistance<T>(bvp.s_radius, bvp.s_x, tid);

      minReduce<T>(bvp.s_radius, dim, tid);
      // local copy of radius
      // TODO: convert local(GLOBAL) copy to register variable.
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

    // TODO eliminate s_result variable?
    // boundary eval
    evaluateBoundary<T>(bvp.s_x, bvp.s_cache, bvp.s_result, dim, tid);
    // return boundary value
    if (threadIdx.x == 0) {

      d_global[blockIdx.x + blockDim.x * i] += bvp.s_result[0];
    }
  }
}

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid) {
  //__syncthreads(); // needed for race check
  int i = blockDim.x / 2;
  while (i != 0) {
    if (tid < i) {
      if (tid + i < dim) {
        if (abs(s_radius[tid]) > abs(s_radius[tid + i]))
          s_radius[tid] = s_radius[tid + i];
      }
    }
    __syncthreads();
    i /= 2;
  }
}

template <typename T>
__device__ void sumReduce(T *s_cache, int tid) {

  // TODO optimize reduce unwraping etc....
  //__syncthreads(); // needed for racecheck
  int i = blockDim.x / 2;
  while (i != 0) {
    if (tid < i) {
      s_cache[tid] += s_cache[tid + i];
    }
    __syncthreads();
    i /= 2;
  }
#ifdef DEBUG
  if (tid == 0)
    printf("%f\n", s_cache[tid]);
  __syncthreads();
#endif
}

template <typename T>
__device__ void warpReduce(T *sdata, int tid) {
  // each thread puts its local sum value into warp variable
  T mySum = sdata[tid];
  unsigned int blockSize = blockDim.x;

  __syncthreads();

  // do reduction in shared mem
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

  // write result for this block to global mem
  if (tid == 0)
    sdata[0] = mySum;

  __syncthreads();
}

template <typename T>
__device__ void broadcast(T *s_radius, int tid) {

  // if (tid != 0) // needed for race condition check
  s_radius[tid] = s_radius[0];
  //__syncthreads(); // needed for race condition check
}

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid) {
  boundaryDistance(cache, s_x, tid);
  minReduce(cache, dim, tid);
  broadcast(cache, tid);

  s_x[tid] = ((1 - (s_x[tid])) == cache[tid]) ? 1 : s_x[tid];
  __syncthreads();
}

template <typename T>
__device__ void norm2(T *s_radius, T *s_cache, int tid) {

  // square vals and copy to cache
  s_cache[tid] = s_radius[tid] * s_radius[tid];
  // sumReduce(s_cache, tid);
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
  s_radius[tid] = s_radius[tid] / cache[0];
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
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *s_result,
                                 const size_t dim, int tid) {

  s_cache[tid] = s_x[tid] * s_x[tid];

  // sumReduce(s_cache, tid);
  warpReduce<T>(s_cache, tid);

  if (tid == 0) {
    s_result[0] = s_cache[0] / (2 * dim);
  }
  __syncthreads();
}

//==============================================================================
template <typename T>
void wos(unsigned int blocks, size_t threads, T *d_x0, T *d_runs, T d_eps,
         const int dim, unsigned int runsperblock, size_t smemSize) {

  printf("setting up problem\n");
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
