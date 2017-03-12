#include "params.h"
#include "helper.h"
#include <cmath>

#include <cuda_runtime.h>

void Parameters::updateNumThreads() {
  numThreads = (isPow2(x0.dimension)) ? x0.dimension : nextPow2(x0.dimension);
}

void Parameters::updateEps() {
  // TODO different strategies for eps
  // eps = 1.0 / (float)sqrt(x0.dimension);
  // eps = 0.01;
}

void Parameters::updateSizeSharedMemory() {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  size_SharedMemory = (4 * numThreads + 1) * sizeof(float);
}

void Parameters::outputParameters(int count) {
  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%ld\n\tnumber of blocks:\t%d\n"
         "\tIterations per blocks:\t%d\n\tremainder per "
         "blocks:\t%d\n\tnumThreads:\t\t%d\n\teps:\t\t\t%f\n",
         x0.value, x0.dimension, totalPaths, numberBlocks, blockIterations,
         blockRemainder, numThreads, eps);
}

void Parameters::updatePathsPerBlock() {

  // source::https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0); // assume one device for now

  if (totalPaths <= MAX_BLOCKS) {
    numberBlocks = totalPaths;
    blockIterations = 1;
    blockRemainder = numberBlocks;
  } else {
    numberBlocks = MAX_BLOCKS;
    blockIterations = ceil((totalPaths / (float)MAX_BLOCKS));
    blockRemainder = totalPaths % MAX_BLOCKS;
  }
}

void Parameters::updateNumBlocksAndThreads() {

  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop; // for more than one graphics card

  updatePathsPerBlock();
}
