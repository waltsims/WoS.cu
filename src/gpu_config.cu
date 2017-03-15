
#include "gpu_config.h"
#include "helper.h"
#include "helper_cuda.h"

GPUConfig GPUConfig::createConfig(Parameters p) {
  unsigned long int gpuPaths;
  int blockIterations;
  unsigned int numThreads;
  int blockRemainder;
  unsigned int numberBlocks;
  int nGPU;
  size_t size_SharedMemory;

  // updaate numThreads
  numThreads =
      (isPow2(p.x0.dimension)) ? p.x0.dimension : nextPow2(p.x0.dimension);
  // update size of shared memory
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  size_SharedMemory = (4 * numThreads + 1) * sizeof(float);

  // TODO: set GPUpaths, nGPU,
  checkCudaErrors(cudaGetDeviceCount(&nGPU));
  gpuPaths = p.totalPaths / nGPU;

  // uppdate paths per block
  if (gpuPaths <= MAX_BLOCKS) {
    numberBlocks = gpuPaths;
    blockIterations = 1;
    blockRemainder = numberBlocks;
  } else {
    numberBlocks = MAX_BLOCKS;
    blockIterations = ceil((gpuPaths / (float)MAX_BLOCKS));
    blockRemainder = gpuPaths % MAX_BLOCKS;
  }

  return GPUConfig(numThreads, gpuPaths, blockIterations, blockRemainder,
                   numberBlocks, nGPU, size_SharedMemory);
}
