#include "params.h"
#include "helper.h"
#include <cmath>

#include <cuda_runtime.h>

void Parameters::updateNumThreads() {
  wos.numThreads = (isPow2(wos.x0.dimension)) ? wos.x0.dimension
                                              : nextPow2(wos.x0.dimension);
}

void Parameters::updateEps() {
  // TODO different strategies for eps
  // wos.eps = 1.0 / (float)sqrt(wos.x0.dimension);
  // wos.eps = 0.01;
}

void Parameters::updateSizeSharedMemory() {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  wos.size_SharedMemory = (4 * wos.numThreads + 1) * sizeof(float);
}

void Parameters::outputParameters(int count) {
  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%ld\n\tnumber of blocks:\t%d\n"
         "\tIterations per blocks:\t%d\n\tremainder per "
         "blocks:\t%d\n\tnumThreads:\t\t%d\n\teps:\t\t\t%f\n",
         wos.x0.value, wos.x0.dimension, wos.totalPaths, wos.numberBlocks,
         wos.blockIterations, wos.blockRemainder, wos.numThreads, wos.eps);
}

void Parameters::updatePathsPerBlock() {

  // source::https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0); // assume one device for now

  if (wos.totalPaths <= MAX_BLOCKS) {
    wos.numberBlocks = wos.totalPaths;
    wos.blockIterations = 1;
    wos.blockRemainder = wos.numberBlocks;
  } else {
    wos.numberBlocks = MAX_BLOCKS;
    wos.blockIterations = ceil((wos.totalPaths / (float)MAX_BLOCKS));
    wos.blockRemainder = wos.totalPaths % MAX_BLOCKS;
  }
}

void Parameters::updateNumBlocksAndThreads() {

  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop; // for more than one graphics card

  updatePathsPerBlock();
}
