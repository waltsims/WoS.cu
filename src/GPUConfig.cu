
#include "GPUConfig.h"
#include "helper.h"

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
  // source::https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
  // uppdate paths per block

  if (p.totalPaths <= MAX_BLOCKS) {
    numberBlocks = p.totalPaths;
    blockIterations = 1;
    blockRemainder = numberBlocks;
  } else {
    numberBlocks = MAX_BLOCKS;
    blockIterations = ceil((p.totalPaths / (float)MAX_BLOCKS));
    blockRemainder = p.totalPaths % MAX_BLOCKS;
  }

  // TODO: set GPUpaths, nGPU,
  nGPU = 1;
  gpuPaths = p.totalPaths;

  return GPUConfig(numThreads, gpuPaths, blockIterations, blockRemainder,
                   numberBlocks, nGPU, size_SharedMemory);
}
