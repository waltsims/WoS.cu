
#include "params.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
// Source: http://www.cplusplus.com/articles/DEN36Up4/

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction
// kernel
//   For kernel 6, we observe the maximum specified number of blocks, because
//   each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
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

static void show_usage(char *argv[]) {
  // TODO: update usage
  std::cerr
      << "Usage: " << argv[0] << " <option(s)> SOURCES"
      << "Options:\n"
      << "\t-h,\t--help\t\t\t\tShow this help message\n"
      << "\t-nbr,\t--numBlocks <NUMBLOCKS>\t\tspecify number of blocks for "
         "reduction\n"
      << "\t-ntr,\t--numThreads <NUMTHREADS>\tspecify number of threads "
         "for reduction\n"
      << "\t-x0,\t--x0Value\t\t\tdefine consant value for x0\n"
      << "\t-dim,\t--dimension\t\t\tdefine the dimension of the problem. "
         "(ergo length of vector x0)\n"
      << "\t-it,\t--iterations\t\t\tdefine the number of iterations for the "
         "algorithm.\n"
      << std::endl;
}

int parseParams(int argc, char *argv[], Parameters &p) {
  // set default values
  unsigned int count = 0;
  // p.reduction.threads = 256; // is curently overwriten
  // p.reduction.blocks = 2;    // is curently overwriten
  p.wos.x0.value = 0.0; // another option for default val is 1.0
  p.wos.x0.dimension = 512;
  p.wos.iterations = 65535;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      show_usage(argv);
      return 0;
    } else if ((arg == "-nbr") || (arg == "--numBlocks")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.reduction.blocks = atoi(argv[i]);
      } else {
        std::cerr << "--numBlocks option requires one argument." << std::endl;
        return 0;
      }
    } else if ((arg == "-ntr") || (arg == "--numThreads")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.reduction.threads = atoi(argv[i]);
      } else {
        std::cerr << "--numThreads option requires one argument." << std::endl;
        return 0;
      }
    } else if ((arg == "-x0") || (arg == "--x0Value")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.wos.x0.value = atof(argv[i]);
      } else {
        std::cerr << "--x0Value option requires one argument." << std::endl;
        return 0;
      }
    } else if ((arg == "-dim") || (arg == "--dimension")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.wos.x0.dimension = atoi(argv[i]);
      } else {
        std::cerr << "--dimension option requires one argument." << std::endl;
        return 0;
      }
    } else if ((arg == "-it") || (arg == "--iterations")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.wos.iterations = atoi(argv[i]);
      } else {
        std::cerr << "--itterations option requires one argument." << std::endl;
        return 0;
      }
    } else {
      show_usage(argv);
      return 0;
    }
  }

  getNumBlocksAndThreads(p.wos.iterations, 65535, 1024, p.reduction.blocks,
                         p.reduction.threads);

  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "iterations:\t\t%d\n\tReduction blocks:\t%d\n\tReductions "
         "threads:\t%d\n\n",
         p.wos.x0.value, p.wos.x0.dimension, p.wos.iterations,
         p.reduction.blocks, p.reduction.threads);
  return 1;
}
