#include <curand_kernel.h>
#include <iostream>

#include "reduce_kernel.cuh"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

extern "C" bool isPow2(unsigned int x);

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x);

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid);

template <typename T>
__device__ void broadcast(T *s_radius, int tid);

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid);

template <typename T>
__device__ void sumReduce(T *s_cache, int tid);

template <typename T>
__device__ void norm2(T *s_radius, int tid);

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid);

template <typename T>
__device__ T boundaryDistance(T d_x, size_t dim, int tid);

template <typename T>
__device__ void evaluateBoundary(T *s_x, T *s_cache, T *s_result,
                                 const size_t dim, int tid);

// TODO ask stack exchange question
// why can't this structure template be a device structure?
template <typename T>
struct BlockVariablePointers {
  T *s_radius, *s_direction, *s_cache, *s_x, *s_result;
};

// TODO make class that holds both structure and function
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
size_t getSizeSharedMem(size_t len) {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  return (4 * len + 1) * sizeof(T);
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

        bvp.s_radius[tid] = boundaryDistance<T>(bvp.s_x[tid], dim, tid);
        __syncthreads();

        minReduce<T>(bvp.s_radius, dim, tid);
        // local register copy of radius
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
        // TODO for runcount indipendent of number of blocks
        // atomicAdd(d_runs, 1);
      }
      __syncthreads();
    }
  }
}

template <typename T>
__device__ void minReduce(T *s_radius, size_t dim, size_t tid) {
  int i = blockDim.x / 2;
  while (i != 0) {
    if (tid < i) {
      if (tid + i < dim) {
        s_radius[tid] = (abs(s_radius[tid]) < abs(s_radius[tid + i]))
                            ? s_radius[tid]
                            : s_radius[tid + i];
      } else {
        s_radius[tid] = s_radius[tid];
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
  __syncthreads();
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
__device__ void norm2(T *s_radius, int tid) {

  // square vals
  s_radius[tid] *= s_radius[tid];
  __syncthreads();
  sumReduce(s_radius, tid);

  if (threadIdx.x == 0) {
    s_radius[0] = sqrt(s_radius[0]);
#ifdef DEBUG
    printf("the 2norm of the value r is %f\n", s_radius[0]);
#endif
  }
  __syncthreads();
}

template <typename T>
__device__ void normalize(T *s_radius, T *cache, size_t dim, int tid) {

  // TODO: does every thread need to do this calculation?
  // or are device calls per thread basis
  cache[tid] = s_radius[tid];
  norm2(cache, tid);
  if (tid < dim)
    s_radius[tid] = s_radius[tid] / cache[0];
  __syncthreads();
#ifdef DEBUG
  printf("normalized value on thread %d after normilization: %f \n",
         threadIdx.x, s_radius[tid]);
#endif
}

template <typename T>
__device__ T boundaryDistance(T d_x, size_t dim, int tid) {
  return (tid < dim) ? 1.0 - abs(d_x) : 0.0;
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
