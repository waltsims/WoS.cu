#include "params.h"

void Parameters::updateLength() {
  wos.x0.length = (isPow2(wos.x0.dimension)) ? wos.x0.dimension
                                             : nextPow2(wos.x0.dimension);
}

void Parameters::updateSizeSharedMemory() {

  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  wos.size_SharedMemory = (4 * wos.x0.length) * sizeof(double);
}

void Parameters::outputParameters(int count) {
  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%d\n\tReduction blocks:\t%d\n\tReductions "
         "threads:\t%d\n",
         wos.x0.value, wos.x0.dimension, wos.totalPaths, reduction.blocks,
         reduction.threads);
}

void Parameters::updatePathsPerBlock() {
  // TODO optimize
  unsigned int &pathsPerBlock = wos.pathsPerBlock;
  unsigned int &number_blocks = wos.totalPaths;
  unsigned int paths = wos.totalPaths;

  if (paths > MAX_BLOCKS) {
    while (MAX_BLOCKS < number_blocks) {
      pathsPerBlock++;
      number_blocks /= pathsPerBlock;
    }
    printf("runs: %d\nnumber of blocks: %d \n runs per block: %d\n", paths,
           number_blocks, pathsPerBlock);
  }
}

void Parameters::updateNumBlocksAndThreads() {

  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int n = wos.totalPaths;
  int maxBlocks = MAX_BLOCKS;
  int maxThreads = MAX_THREADS;
  int &blocks = reduction.blocks;
  int &threads = reduction.threads;

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
