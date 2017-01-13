#include <curand_kernel.h>
#include <iostream>

#include "params.h"
#include "reduce_kernel.cuh"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid);

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
__device__ void boundaryDistance(T *s_radius, T *d_x, size_t dim, int tid);

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
  bvp.s_x[tid] = 0.0;
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
    if (tid < len) {
      // seed for random number generation
      unsigned int seed = index;
      curandState s;
      curand_init(seed, 0, 0, &s);

      // TODO: x0 in constmem

      T r = INFINITY;
      // max step size
      while (d_eps < r) {

        boundaryDistance<T>(bvp.s_radius, bvp.s_x, dim, tid);
        __syncthreads();

        minReduce<T>(bvp.s_radius, dim, tid);
        // local copy of radius
        // TODO: convert local(GLOBAL) copy to register variable.
        r = bvp.s_radius[0];

        // random next step_direction
        bvp.s_direction[tid] = (tid < dim) ? curand_normal(&s) : 0.0;

        // normalize direction with L2 norm
        normalize<T>(bvp.s_direction, bvp.s_cache, dim, tid);

        // next x point
        bvp.s_x[tid] += r * bvp.s_direction[tid];
      }

      // find closest boundary point
      round2Boundary<T>(bvp.s_x, bvp.s_cache, dim, tid);

      // TODO eliminate s_reslt variable?
      // boundary eval
      evaluateBoundary<T>(bvp.s_x, bvp.s_cache, bvp.s_result, dim, tid);
      // return boundary value
      if (threadIdx.x == 0) {

        d_global[blockIdx.x + blockDim.x * i] += bvp.s_result[0];

        // d_global[blockIdx.x] = s_x[0];
        // TODO for runcount independant of number of blocks
        // atomicAdd(d_runs, 1);
      }
      //      __syncthreads(); // doesn't seem neccisarry
    }
  }
}

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid) {
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
__device__ void broadcast(T *s_radius, int tid) {

  if (tid != 0)
    s_radius[tid] = s_radius[0];
  __syncthreads();
}

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid) {
  cache[tid] = 1 - abs(s_x[tid]);
  __syncthreads(); // no noticable effect on accuracy but leads to race
  // condition
  minReduce(cache, dim, tid);
  broadcast(cache, tid);

  s_x[tid] = ((1 - (s_x[tid])) == cache[tid]) ? 1 : s_x[tid];
}

template <typename T>
__device__ void sumReduce(T *s_cache, int tid) {

  // TODO optimize reduce unwraping etc....
  __syncthreads();
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
#endif
}

template <typename T>
__device__ void norm2(T *s_radius, T *s_cache, int tid) {

  // square vals and copy to cache
  s_cache[tid] = s_radius[tid] * s_radius[tid];
  __syncthreads();
  sumReduce(s_cache, tid);

  if (threadIdx.x == 0) {
    s_cache[0] = sqrt(s_cache[0]);
#ifdef DEBUG
    printf("the 2norm of the value r is %f\n", s_cache[0]);
#endif
  }
  __syncthreads();
}

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid) {

  norm2(s_radius, cache, tid);
  if (tid < dim)
    s_radius[tid] = s_radius[tid] / cache[0];
  __syncthreads();
#ifdef DEBUG
  printf("normalized value on thread %d after normilization: %f \n",
         threadIdx.x, s_radius[tid]);
#endif
}

template <typename T>
__device__ void boundaryDistance(T *s_radius, T *d_x, size_t dim, int tid) {
  s_radius[tid] = (tid < dim) ? 1.0 - abs(d_x[tid]) : 0.0;
}

template <typename T>
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *s_result,
                                 const size_t dim, int tid) {

  s_cache[tid] = s_x[tid] * s_x[tid];

  sumReduce(s_cache, tid);
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
