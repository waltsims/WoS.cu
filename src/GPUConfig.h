#ifndef GPUCONFIG_H
#define GPUCONFIG_H
#include "params.h"

class GPUConfig {
public:
  const unsigned long int gpuPaths;
  const int blockIterations;
  const unsigned int numThreads;
  const int blockRemainder;
  const unsigned int numberBlocks;
  const int nGPU;
  const size_t size_SharedMemory;

  static GPUConfig createConfig(Parameters p);

private:
  GPUConfig(const unsigned int numThreads, const unsigned long int gpuPaths,
            const int blockIterations, const int blockRemainder,
            const unsigned int numberBlocks, const int nGPU,
            const size_t size_SharedMemory)
      : numThreads(numThreads), gpuPaths(gpuPaths),
        blockIterations(blockIterations), blockRemainder(blockRemainder),
        numberBlocks(numberBlocks), nGPU(nGPU),
        size_SharedMemory(size_SharedMemory) {}
};

#endif // GPUCONFIG_H
