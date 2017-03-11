#include "params.h"
#include "helper.hpp"
#include <cmath>

#include <cuda_runtime.h>

void Parameters::updateNumThreads() {
  wos.numThreads = (isPow2(wos.x0.dimension)) ? wos.x0.dimension
                                              : nextPow2(wos.x0.dimension);
}

// void Parameters::updateNumThreads() {
//
//   if (wos.x0.dimension < 32) {
//     wos.numThreads = 32;
//   } else {
//     int mod = wos.x0.dimension % 32;
//     wos.numThreads =
//         (mod == 0) ? wos.x0.dimension : (wos.x0.dimension + (32 - mod));
//   }
// }
void Parameters::updateEps() {
  // TODO different strategies for eps
  // wos.eps = 1.0 / (float)sqrt(wos.x0.dimension);
  wos.eps = 0.01;
}

void Parameters::updateSizeSharedMemory() {

  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  wos.size_SharedMemory = (4 * wos.x0.dimension + 1) * sizeof(double);
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

  // wos.pathsPerBlock = 1000;
  // wos.numberBlocks = 10000;
  // wos.totalPaths = wos.pathsPerBlock * wos.numberBlocks;

  // wos.pathsPerBlock = 1;
  // wos.numberBlocks = wos.totalPaths;
}

void Parameters::updateNumBlocksAndThreads() {

  // get device capability, to avoid block/grid size exceed the upper bound

  // dead code for reduce
  cudaDeviceProp prop;
  int maxBlocks = MAX_BLOCKS;
  int maxThreads = MAX_THREADS;
  int &blocks = reduction.blocks;
  int &threads = reduction.threads;

  updatePathsPerBlock();

  // int device;
  // cudaGetDevice(&device);
  // cudaGetDeviceProperties(&prop, device);
  //
  // // TODO implicit type conversion with nextPow2 to int
  // TODO fix reduce if neccisarry.
  // threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
  // blocks = (n + (threads * 2 - 1)) / (threads * 2);
  //
  // if ((float)threads * blocks >
  //     (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
  //   printf("n is too large, please choose a smaller number!\n");
  // }
  //
  // if (blocks > prop.maxGridSize[0]) {
  //   printf("Grid size <%d> exceeds the device capability <%d>, set block
  //   size
  //   "
  //          "as %d (original %d)\n",
  //          blocks, prop.maxGridSize[0], threads * 2, threads);
  //
  //   blocks /= 2;
  //   threads *= 2;
  // }
  //
  // blocks = (maxBlocks < blocks) ? maxBlocks : blocks;
}
