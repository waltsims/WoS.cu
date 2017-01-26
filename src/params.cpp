#include "params.h"

size_t getLength(size_t dim) { return (isPow2(dim)) ? dim : nextPow2(dim); }

template <bool isDouble>
size_t getSizeSharedMem(size_t len);

template <>
size_t getSizeSharedMem<true>(size_t len) {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  return (4 * len) * sizeof(double);
}

template <>
size_t getSizeSharedMem<false>(size_t len) {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  return (4 * len) * sizeof(float);
}

unsigned int getPathsPerBlock(unsigned int runs, unsigned int &number_blocks) {
  unsigned int pathsPerBlock = 1;
  number_blocks = runs;
  if (runs > MAX_BLOCKS) {
    while (MAX_BLOCKS < number_blocks) {
      pathsPerBlock++;
      number_blocks /= pathsPerBlock;
    }
    printf("runs: %d\nnumber of blocks: %d \n runs per block: %d\n", runs,
           number_blocks, pathsPerBlock);
  }
  return pathsPerBlock;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {

  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  // TODO implicit type conversion with nextPow2 to int
  threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
  blocks = (n + (threads * 2 - 1)) / (threads * 2);

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf("Grid size <%d> exceeds the device capability <%d>, set block size "
           "as %d (original %d)\n",
           blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }

  blocks = (maxBlocks < blocks) ? maxBlocks : blocks;
}
