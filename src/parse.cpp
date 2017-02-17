#include "helper.hpp"
#include "params.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
      << "\t-st,\t--simulation-type\t\t[thrust | native | host]\n"
      << std::endl;
}

int parseParams(int argc, char *argv[], Parameters &p) {
  // set default values
  unsigned int count = 0;
  // p.reduction.threads = 256; // is curently overwriten
  // p.reduction.blocks = 2;    // is curently overwriten

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
    } else if ((arg == "-st") || (arg == "--simulation-type")) {
      if (i + 1 < argc) {
        i++;
        count++;
        if (!strcmp(argv[i], "thrust"))
          p.wos.simulation = thrustWos;
        else if (!strcmp(argv[i], "native"))
          p.wos.simulation = nativeWos;
        else if (!strcmp(argv[i], "host"))
          p.wos.simulation = hostWos;
        else {
          std::cerr
              << "--simulation-type option requires one argument.[thrust | "
                 "native]"
              << std::endl;
          return 0;
        }
      } else {
        std::cerr << "--simulation-type option requires one argument.[thrust | "
                     "native]"
                  << std::endl;
        return 0;
      }
    } else {
      show_usage(argv);
      return 0;
    }
  }

  p.update();
  p.outputParameters(count);

  return 1;
}
