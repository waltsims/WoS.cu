#include "helper.h"

#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <math_functions.h>
#include <stdio.h>

//#include <cublas_v2.h>

#define MAX_THREADS 1024

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#ifdef DEBUG
bool debug = true;
#endif

// source: CUDA reduction documentation
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

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

template <typename T> __device__ void broadcast(T *s_radius, int tid) {

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

template <typename T> __device__ void sumReduce(T *s_cache, int tid) {

  // TODO optimize reduce unwraping etc....

  int i = blockDim.x / 2;
  while (i != 0) {
    if (tid < i) {
      s_cache[tid] += s_cache[tid + i];
    }
    i /= 2;
  }
  // if (tid == 0)
  //   printf("%f\n", s_cache[tid]);
}

template <typename T> __device__ void norm2(T *s_radius, int tid) {

  // square vals
  s_radius[tid] *= s_radius[tid];

  sumReduce(s_radius, tid);

  if (threadIdx.x == 0) {
    s_radius[0] = sqrt(s_radius[0]);
    //  printf("the 2norm of the value r is %f\n", s_radius[0]);
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
  //  printf("normalized value on thread %d after normilization: %f \n",
  //       threadIdx.x, s_radius[tid]);
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

// source
// reduce6 from cuda documentation
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sumReduce(T *g_idata, T *g_odata, unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
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
    g_odata[blockIdx.x] = mySum;
}

template <typename T>
__global__ void WoS(T *d_x0, T *d_global, T d_eps, size_t dim, size_t len,
                    bool debug) {

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

    // if (debug)
    //   printf("d_x0 on thread %d: %f \n", threadIdx.x, s_x[threadIdx.x]);

    T r = INFINITY;
    // max step size
    while (d_eps < r) {

      s_radius[tid] = boundaryDistance<T>(s_x[tid], dim, tid);

      // if (debug)
      //   printf("s_radius on thread %d: %f \n", threadIdx.x, s_radius[tid]);

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

      // if (debug)
      //   printf("s_direction on thread %d after randomization: %f \n",
      //          threadIdx.x, s_direction[tid]);

      // normalize direction with L2 norm
      normalize<T>(s_direction, s_cache, dim, tid);

      // next x point
      s_x[tid] += r * s_direction[tid];

      // if (debug)

      //  printf("next x on thread %d after step: %f \n", threadIdx.x,
      //  s_x[tid]);
    }

    // find closest boundary point
    round2Boundary<T>(s_x, s_cache, dim, tid);
    // if (debug)
    //   printf("x on thread %d after rounding: %f \n", threadIdx.x,
    //   s_x[tid]);
    // boundary eval
    evaluateBoundary<T>(s_x, s_cache, s_result, dim, tid);

    // return boundary value
    if (threadIdx.x == 0) {

      // if (debug)
      // printf("end %d with result %f on Block %d \n", runCount, s_result[0],
      //            blockIdx.x);

      // atomicAdd(d_global, s_result[0]); //Float alternative
      d_global[blockIdx.x] = s_result[0];
      // TODO for runcount indipendent of number of blocks
      // atomicAdd(d_runs, 1);
    }
    __syncthreads();
  }
  // TODO reduce d_global
}

// source: cuda reduction documentation
template <typename T>
void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata) {

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  if (isPow2(size)) {
    switch (threads) {
    case 512:
      sumReduce<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 256:
      sumReduce<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 128:
      sumReduce<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 64:
      sumReduce<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 32:
      sumReduce<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 16:
      sumReduce<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 8:
      sumReduce<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 4:
      sumReduce<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 2:
      sumReduce<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 1:
      sumReduce<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;
    }
  } else {
    switch (threads) {
    case 512:
      sumReduce<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 256:
      sumReduce<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 128:
      sumReduce<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 64:
      sumReduce<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 32:
      sumReduce<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 16:
      sumReduce<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 8:
      sumReduce<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 4:
      sumReduce<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 2:
      sumReduce<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 1:
      sumReduce<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;
    }
  }
}

template void reduce<float>(int size, int threads, int blocks, float *d_idata,
                            float *d_odata);

template void reduce<double>(int size, int threads, int blocks, double *d_idata,
                             double *d_odata);

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

template <typename T>
void outputConvergence(const char *filename, T *vals, int runs) {

  std::ofstream file(filename);
  file << "run [pow 10]\t"
       << "solution val\t" << std::endl;

  float partSum = 0;
  for (int i = 1; i < runs; i++) {
    partSum += vals[i];
    file << i << "\t" << partSum / i << "\t" << std::endl;
  }
  file.close();
}

main(int argc, char *argv[]) {
  bool debug = true;

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
  const unsigned int runs = 1024 * 10;
  // TODO for runcount indipendent of number of blocks
  // int *d_runs;

  T x0[len];
  T h_results[blocks];
  T h_runs[runs];
  T *d_x0;
  T *d_runs;
  T *d_results;
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
  WoS<T><<<runs, len, (4 * len + 1) * sizeof(T)>>>(d_x0, d_runs, d_eps, dim,
                                                   len, debug);
  cudaDeviceSynchronize();
  computationTime.end();

  // convergence plot export
  cudaStat =
      cudaMemcpy(&h_runs, d_runs, runs * sizeof(T), cudaMemcpyDeviceToHost);
  outputConvergence("cuWos_convergence.dat", h_runs, runs);

  ///////

  reduce(runs, threads, blocks, d_runs, d_results);

  cudaStat = cudaMemcpy(&h_results, d_results, blocks * sizeof(T),
                        cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) {
    printf(" device memory download failed\n");
    return EXIT_FAILURE;
  }

  reduceCPU(h_results, blocks);

  totalTime.end();

  printf("average: %f  \nrunning time: %f sec  \ntotal time: %f sec \n",
         h_results[0] / runs, computationTime.get(), totalTime.get());
  cudaFree(d_results);
  cudaFree(d_x0);

  for (int i = 0; i < argc; i++)
    printf("Argument %d:  %s\n", i, argv[i]);
  return (0);
}
