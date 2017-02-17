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

void Parameters::updateSizeSharedMemory() {

  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  wos.size_SharedMemory = (4 * wos.x0.dimension + 1) * sizeof(double);
}

void Parameters::outputParameters(int count) {
  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%d\n\tnumber of blocks:\t%d\n"
         "\tPaths per blocks:\t%d\n\tnumThreads:\t\t%d\n",
         wos.x0.value, wos.x0.dimension, wos.totalPaths, wos.numberBlocks,
         wos.pathsPerBlock, wos.numThreads);
}

void Parameters::updatePathsPerBlock() {
  // TODO optimize
  unsigned int &pathsPerBlock = wos.pathsPerBlock;
  unsigned int &number_blocks = wos.totalPaths;
  unsigned int paths = wos.totalPaths;

  unsigned int i = floor(sqrt(wos.totalPaths));

  while (wos.totalPaths % i != 0) {
    i--;
  }

  wos.numberBlocks =
      (i < wos.totalPaths / i) ? wos.totalPaths / i : i; // 21845;
  wos.pathsPerBlock = wos.totalPaths / wos.numberBlocks;

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
  //   printf("Grid size <%d> exceeds the device capability <%d>, set block size
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
