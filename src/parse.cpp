#include "helper.hpp"

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
      << "\t-it,\t--totalPaths\t\t\tdefine the number of iterations for the "
         "algorithm.\n"
      << "\t-t,\t--type\t\t\t\ta boolean value for the problem type [1 = "
         "double, "
         "0 = float]\n"
      << std::endl;
}

int parseParams(int argc, char *argv[], Parameters &p) {
  // set default values
  unsigned int count = 0;
  // p.reduction.threads = 256; // is curently overwriten
  // p.reduction.blocks = 2;    // is curently overwriten
  p.wos.x0.value = 0.0; // another option for default val is 1.0
  p.wos.x0.dimension = 512;
  p.wos.totalPaths = 65535;
  p.wos.typeDouble = true;

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
    } else if ((arg == "-it") || (arg == "--totalPaths")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.wos.totalPaths = atoi(argv[i]);
      } else {
        std::cerr << "--itterations option requires one argument." << std::endl;
        return 0;
      }
    } else if ((arg == "-t") || (arg == "--type")) {
      if (i + 1 < argc) {
        i++;
        count++;
        p.wos.typeDouble = (argv[i] == "1");
      } else {
        std::cerr << "--itterations option requires one argument." << std::endl;
        return 0;
      }
    } else {
      show_usage(argv);
      return 0;
    }
  }

  getNumBlocksAndThreads(p.wos.totalPaths, 65535, 1024, p.reduction.blocks,
                         p.reduction.threads);

  p.wos.x0.length =
      getLength(p.wos.x0.dimension); // length of the storage vector

  if (p.wos.typeDouble)
    p.wos.size_SharedMemory = getSizeSharedMem<true>(p.wos.x0.length);
  else
    p.wos.size_SharedMemory = getSizeSharedMem<false>(p.wos.x0.length);

  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%d\n\tReduction blocks:\t%d\n\tReductions "
         "threads:\t%d\n\tvariable type:\t\t%s\n\n",
         p.wos.x0.value, p.wos.x0.dimension, p.wos.totalPaths,
         p.reduction.blocks, p.reduction.threads,
         (p.wos.typeDouble) ? "double" : "float");
  return 1;
}
