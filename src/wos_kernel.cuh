#include <curand_kernel.h>
#include <iostream>

#include "reduce_kernel.cuh"

#define MAX_THREADS 1024

extern "C" bool isPow2(unsigned int x);

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x);

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

  // TODO: this doesn't need to look like this:
  // solution idea s_radius[tid] = s_radius[0]
  int i = 1;
  while (i < blockDim.x) {
    if (threadIdx.x < i) {
      s_radius[tid + i] = s_radius[tid];
    }
    __syncthreads();
    i *= 2;
  }
}

template <typename T>
__device__ void round2Boundary(T *s_x, T *cache, size_t dim, int tid) {
  cache[tid] = 1 - abs(s_x[tid]);
  minReduce(cache, dim, tid);
  broadcast(cache, tid);

  s_x[tid] = ((1 - (s_x[tid])) == cache[tid]) ? 1 : s_x[tid];
}

template <typename T>
__device__ void sumReduce(T *s_cache, int tid) {

  // TODO optimize reduce unwraping etc....

  int i = blockDim.x / 2;
  while (i != 0) {
    if (tid < i) {
      s_cache[tid] += s_cache[tid + i];
    }
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

  // TODO: better implimentation of sum reduce would be better
  s_cache[tid] = s_x[tid] * s_x[tid];

  sumReduce(s_cache, tid);
  if (tid == 0) {
    s_result[0] = s_cache[0] / (2 * dim);
  }
  __syncthreads();
}

template <typename T>
__global__ void WoS(T *d_x0, T *d_global, T d_eps, size_t dim, size_t len) {

#ifdef DEBUG
  printf("Wos called!\n");
#endif

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // shared buff should currently be 4 * problem size + 1
  // TODO reduce the number of shared variabls
  extern __shared__ T buff[];

  // TODO make this into struct for data coelecing?
  T *s_radius = buff;
  T *s_direction = len + buff;
  T *s_cache = len + s_direction;
  T *s_x = len + s_cache;

  // Put this last because long term goal is constant memory.
  T *s_result = len + s_x;
  if (tid == 0)
    s_result[0] = 0.0;
  __syncthreads();

  if (tid < len) {
    // seed for random number generation
    unsigned int seed = index;
    curandState s;
    curand_init(seed, 0, 0, &s);

    // copy x0 to local __shared__"moveable" x
    // TODO: x0 in constmem
    s_radius[tid] = INFINITY;
    s_x[tid] = d_x0[tid];

#ifdef DEBUG
    printf("d_x0 on thread %d: %f \n", threadIdx.x, s_x[threadIdx.x]);
#endif

    T r = INFINITY;
    // max step size
    while (d_eps < r) {

      s_radius[tid] = boundaryDistance<T>(s_x[tid], dim, tid);

#ifdef DEBUG
      printf("s_radius on thread %d: %f \n", threadIdx.x, s_radius[tid]);
#endif

      // TODO working minReduce or s_radius value!
      minReduce<T>(s_radius, dim, tid);
      // local register copy of radius
      r = s_radius[0];

      // if (threadIdx.x == 0 && debug) {
      //   printf("the minimum of the value r on itteration %d is %f\n",
      //   runCount,
      //          s_radius[0]);
      // }

      // if (debug)
      //   printf("s_radius on thread %d after broadcast: %f \n", threadIdx.x,
      //   r);

      // random next step_direction

      s_direction[tid] = (tid < dim) ? curand_normal(&s) : 0.0;

#ifdef DEBUG
      printf("s_direction on thread %d after randomization: %f \n", threadIdx.x,
             s_direction[tid]);
#endif

      // normalize direction with L2 norm
      normalize<T>(s_direction, s_cache, dim, tid);

      // next x point
      s_x[tid] += r * s_direction[tid];

#ifdef DEBUG
      printf("next x on thread %d after step: %f \n", threadIdx.x, s_x[tid]);
#endif
    }

    // find closest boundary point
    round2Boundary<T>(s_x, s_cache, dim, tid);
#ifdef DEBUG
    printf("x on thread %d after rounding: %f \n", threadIdx.x, s_x[tid]);
#endif
    // boundary eval
    evaluateBoundary<T>(s_x, s_cache, s_result, dim, tid);

    // return boundary value
    if (threadIdx.x == 0) {

#ifdef DEBUG
// printf("end %d with result %f on Block %d \n", runCount, s_result[0],
//        blockIdx.x);
#endif

      d_global[blockIdx.x] = s_result[0];
      // TODO for runcount indipendent of number of blocks
      // atomicAdd(d_runs, 1);
    }
    __syncthreads();
  }
}
