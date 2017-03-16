#include <curand_kernel.h>
#include <iostream>
#include <numeric>
#include <vector>

#include "cpu_reduce.h"
#include "gpu_config.h"
#include "parameters.h"

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

__device__ void warpMinReduce(float *sdata, int tid);

__device__ void warpSumReduce(float *sdata, int tid);

__device__ void broadcast(float *sdata, int tid);

__device__ void project2Boundary(float *s_x, float *cache, size_t dim, int tid);

__device__ void norm2(float *s_direction, float *s_cache, int tid);

__device__ void getBoundaryDistance(float *s_radius, float *d_x, int tid);

__device__ void evaluateBoundaryValue(float *s_x, float *s_cache,
                                      float &s_result, const size_t dim,
                                      int tid);

struct BlockVariablePointers {
  float *s_radius, *s_direction, *s_cache, *s_x, *s_result;
};

__device__ void calcSubPointers(BlockVariablePointers *bvp, size_t len,
                                float *buff) {
  bvp->s_radius = buff;
  bvp->s_direction = len + buff;
  bvp->s_cache = 2 * len + buff;
  bvp->s_x = 3 * len + buff;
  bvp->s_result = 4 * len + buff;
}

__device__ void smemInit(BlockVariablePointers bvp, float *d_x0, int tid) {
  // initialize shared memory
  bvp.s_direction[tid] = 0.0;
  bvp.s_cache[tid] = 0.0;
  bvp.s_radius[tid] = INFINITY;
  // copy x0 to local __shared__"moveable" x
  bvp.s_x[tid] = d_x0[tid];
  if (threadIdx.x == 0)
    bvp.s_result[0] = 0.0;
}

__global__ void WoS(float *d_x0, float *d_global, float *d_avgPath,
                    unsigned int *d_stepCount, float d_eps, size_t dim,
                    bool avgPath, int blockIterations, int blockRemainder,
                    int gpu) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // seed for random number generation
  curandState s;
  unsigned int seed = index * gpu;
  curand_init(seed, 0, 0, &s);

  extern __shared__ float buff[];
  BlockVariablePointers bvp;
  calcSubPointers(&bvp, blockDim.x, buff);
  smemInit(bvp, d_x0, tid);
  if (tid < dim) {
    float x0 = bvp.s_x[tid]; // only works as long as dim x = dim block
    float r;
    float distance;
    unsigned int stepCount;

    if (tid == 0 && avgPath) {
      distance = 0.0;
      stepCount = 0;
    }

#ifdef DEBUG
    if (tid == 0)
      printf("[WOS]: d_global[%d] before:\t%f\n", blockIdx.x,
             d_global[blockIdx.x]);
#endif

    for (int it = 0; it < blockIterations; it++) {
      if ((blockIdx.x < blockRemainder) || (it < blockIterations - 1)) {
        bvp.s_x[tid] = x0;

        // TODO: x0 in texture meomry
        r = INFINITY;

        // max step size
        while (d_eps < r) {

          getBoundaryDistance(bvp.s_radius, bvp.s_x, tid);

          warpMinReduce(bvp.s_radius, tid);
          // local copy of radius
          r = bvp.s_radius[0];

          // random next step_direction
          bvp.s_direction[tid] = curand_normal(&s);

          // normalize direction with L2 norm
          norm2(bvp.s_direction, bvp.s_cache, tid);

          // next x point
          bvp.s_x[tid] += r * bvp.s_direction[tid];
          if (avgPath && tid == 0) {
            distance += r;
            stepCount++;
          }
        }

        // find closest boundary point
        project2Boundary(bvp.s_x, bvp.s_cache, dim, tid);

        // boundary eval and return do global memory
        evaluateBoundaryValue(bvp.s_x, bvp.s_cache, bvp.s_result[0], dim, tid);
      }
    }
    if (tid == 0) {
      d_global[blockIdx.x] = bvp.s_result[tid];
      if (avgPath) {
        d_avgPath[blockIdx.x] = distance;
        d_stepCount[blockIdx.x] = stepCount;
      }
#ifdef DEBUG
      printf("[WOS]: result on block %d:\n%f\n", blockIdx.x, bvp.s_result[0]);
#endif
    }
    __syncthreads();
  }
}

__device__ void warpMinReduce(float *sdata, int tid) {
  // each thread puts its local value into warp variable
  float myMin = sdata[tid];
  unsigned int blockSize = blockDim.x;

  __syncthreads(); // ensure data is ready

  // do reduction in shared mem
  if ((blockSize == 1024) && (tid < 512)) {
    sdata[tid] = myMin =
        ((abs(myMin)) < abs(sdata[tid + 512])) ? myMin : sdata[tid + 512];
  }
  __syncthreads();

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

__device__ void warpSumReduce(float *sdata, int tid) {
  // each thread puts its local sum value into warp variable
  float mySum = sdata[tid];
  unsigned int blockSize = blockDim.x;

  // do reduction in shared mem

  if ((blockSize == 1024) && (tid < 512)) {
    sdata[tid] = mySum = mySum + sdata[tid + 512];
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

__device__ void broadcast(float *sdata, int tid) {

  if (tid != 0) // needed for race condition check
    sdata[tid] = sdata[0];
  __syncthreads(); // needed for race condition check
}

__device__ void project2Boundary(float *s_x, float *cache, size_t dim,
                                 int tid) {
  getBoundaryDistance(cache, s_x, tid);
  warpMinReduce(cache, tid);
  broadcast(cache, tid);

  // TODO: could be faster if index of min was known
  s_x[tid] = ((1.0 - abs(s_x[tid])) == cache[tid])
                 ? 1.0
                 : s_x[tid]; // poor float comparison. unclear that this is
                             // boundary distance again.
  __syncthreads();
}

__device__ void norm2(float *s_direction, float *s_cache, int tid) {

  // square vals and copy to cache
  s_cache[tid] = s_direction[tid] * s_direction[tid];

  warpSumReduce(s_cache, tid);

  s_direction[tid] /= sqrt(s_cache[0]);
  __syncthreads(); // needed for race check
}

__device__ void getBoundaryDistance(float *s_radius, float *d_x, int tid) {
  s_radius[tid] = 1.0 - abs(d_x[tid]);
}

__device__ void evaluateBoundaryValue(float *s_x, float *s_cache,
                                      float &s_result, const size_t dim,
                                      int tid) {

  s_cache[tid] = s_x[tid] * s_x[tid];

  warpSumReduce(s_cache, tid);

  if (tid == 0) {
#ifdef DEBUG
    printf("[WOS]: output from block %d:\t%f\n", blockIdx.x,
           s_cache[0] / (2 * dim));
#endif
    s_result += s_cache[0] / (2 * dim);
  }
  __syncthreads();
}

//==============================================================================
void initX0(float *x0, size_t dim, size_t len, float val);

typedef struct {
  float *d_x0;
  float *d_paths;
  float *d_avgPath;
  unsigned int *d_stepCount;
  cudaStream_t stream;
} MultiGPU;

float wosNative(Timers &timers, Parameters &p, GPUConfig gpu, DataLogger &dl) {

  // declare local array variabls
  float *h_x0;
  checkCudaErrors(
      cudaMallocHost((void **)&h_x0, sizeof(float) * gpu.numThreads));
  // declare pointers for device variables
  MultiGPU multiGPU[gpu.nGPU];

  // init our point on host
  initX0(h_x0, p.x0.dimension, gpu.numThreads, p.x0.value);

  timers.memorySetupTimer.start();

  if (p.verbose) {
    printInfo("initializing d_paths");
  }

  for (int i = 0; i < gpu.nGPU; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaStreamCreate(&multiGPU[i].stream));

    // maloc device memory
    checkCudaErrors(
        cudaMalloc((void **)&multiGPU[i].d_x0, gpu.numThreads * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&multiGPU[i].d_paths,
                               gpu.numberBlocks * sizeof(float)));

    if (p.avgPath) {
      checkCudaErrors(cudaMalloc((void **)&multiGPU[i].d_avgPath,
                                 gpu.numberBlocks * sizeof(float)));

      checkCudaErrors(cudaMalloc((void **)&multiGPU[i].d_stepCount,
                                 gpu.numberBlocks * sizeof(unsigned int)));
    }

    // Let's bring our data to the Device
    checkCudaErrors(
        cudaMemcpyAsync(multiGPU[i].d_x0, h_x0, gpu.numThreads * sizeof(float),
                        cudaMemcpyHostToDevice, multiGPU[i].stream));
  }

  timers.memorySetupTimer.end();
  timers.computationTimer.start();
  if (p.verbose) {
    printInfo("setting up problem");
  }

  if (p.verbose) {
    printInfo("running simulation");
  }
  for (int i = 0; i < gpu.nGPU; i++) {
    checkCudaErrors(cudaSetDevice(i));

    dim3 dimBlock(gpu.numThreads, 1, 1);
    dim3 dimGrid(gpu.numberBlocks, 1, 1);

    cudaError err;

    WoS<<<dimGrid, dimBlock, gpu.size_SharedMemory, multiGPU[i].stream>>>(
        multiGPU[i].d_x0, multiGPU[i].d_paths, multiGPU[i].d_avgPath,
        multiGPU[i].d_stepCount, p.eps, p.x0.dimension, p.avgPath,
        gpu.blockIterations, gpu.blockRemainder, i + 1);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      printf("Wos Kernel returned an error on device %d:\n %s\n", i,
             cudaGetErrorString(err));
    }
  }

  cudaFreeHost(h_x0);
  timers.computationTimer.end();

  // We don't need d_x0 anymore, only to reduce solution data

  if (p.verbose) {
    printInfo("downloading path data");
  }
  timers.memoryDownloadTimer.start();
  float *h_paths = (float *)malloc(sizeof(float) * gpu.nGPU * gpu.numberBlocks);
  float *h_avgPaths;
  std::vector<unsigned int> h_stepCount(gpu.nGPU * gpu.numberBlocks);
  if (p.avgPath) {
    h_avgPaths = (float *)malloc(sizeof(float) * gpu.nGPU * gpu.numberBlocks);
  }

  for (int i = 0; i < gpu.nGPU; i++) {
    cudaFree(multiGPU[i].d_x0);
    checkCudaErrors(cudaSetDevice(i));
    // Download paths data
    checkCudaErrors(
        cudaMemcpyAsync(h_paths + i * gpu.numberBlocks, multiGPU[i].d_paths,
                        gpu.numberBlocks * sizeof(float),
                        cudaMemcpyDeviceToHost, multiGPU[i].stream));
    cudaFree(multiGPU[i].d_paths);
    if (p.avgPath) {
      checkCudaErrors(cudaMemcpyAsync(
          h_avgPaths + i * gpu.numberBlocks, multiGPU[i].d_avgPath,
          gpu.numberBlocks * sizeof(float), cudaMemcpyDeviceToHost,
          multiGPU[i].stream));
      checkCudaErrors(cudaMemcpyAsync(
          h_stepCount.data() + i * gpu.numberBlocks, multiGPU[i].d_stepCount,
          gpu.numberBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost,
          multiGPU[i].stream));
      cudaFree(multiGPU[i].d_avgPath);
    }
  }
  cudaDeviceSynchronize();
  timers.memoryDownloadTimer.end();

  if (p.verbose) {
    printInfo("reduce data on CPU");
  }

  float gpu_result =
      reduceCPU(h_paths, gpu.nGPU * gpu.numberBlocks) / p.totalPaths;
  if (p.avgPath) {
    float avgPathDistance =
        reduceCPU(h_avgPaths, gpu.nGPU * gpu.numberBlocks) / p.totalPaths;
    dl.setAvgPathSet(avgPathDistance);
    float avgStepCount =
        (float)std::accumulate(h_stepCount.begin(), h_stepCount.end(), 0.0) /
        p.totalPaths;
    dl.setAvgNumSteps(avgStepCount);
  }
  free(h_paths);

#ifdef DEBUG
  printf("[MAIN]: results values after copy:\n");
  for (unsigned int n = 0; n < gpu.numberBlocks; n++) {
    printf("%f\n", h_paths[n]);
  }
#endif
  dl.setResult(gpu_result);

  return gpu_result;
}

void initX0(float *h_x0, size_t dim, size_t len, float val) {
  // init our point on host
  for (unsigned int i = 0; i < dim; i++)
    // h_x0[i] = i == 1 ? 0.22 : 0;
    h_x0[i] = val;
  for (unsigned int i = dim; i < len; i++)
    h_x0[i] = 0.0;
}
