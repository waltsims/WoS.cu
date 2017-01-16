
#include "params.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// Source: http://www.cplusplus.com/articles/DEN36Up4/

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
      << "\t-it,\t--itterations\t\t\tdefine the number of itterations for the "
         "algorithm.\n"
      << std::endl;
}

int parseParams(int argc, char *argv[], Parameters &p) {
  // set default values
  p.reduction.threads = 512;
  p.reduction.blocks = 256;
  p.wos.x0.value = 0.0; // another option for default val is 1.0
  p.wos.x0.dimension = 512;
  p.wos.itterations = 65535;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      show_usage(argv);
      return 0;
    } else if ((arg == "-nbr") || (arg == "--numBlocks")) {
      if (i + 1 < argc) {
        i++;
        p.reduction.blocks = atoi(argv[i]);
      } else {
        std::cerr << "--numBlocks option requires one argument." << std::endl;
        return 1;
      }
    } else if ((arg == "-ntr") || (arg == "--numThreads")) {
      if (i + 1 < argc) {
        i++;
        p.reduction.threads = atoi(argv[i]);
      } else {
        std::cerr << "--numThreads option requires one argument." << std::endl;
        return 1;
      }
    } else if ((arg == "-x0") || (arg == "--x0Value")) {
      if (i + 1 < argc) {
        i++;
        p.wos.x0.value = atof(argv[i]);
      } else {
        std::cerr << "--x0Value option requires one argument." << std::endl;
        return 1;
      }
    } else if ((arg == "-dim") || (arg == "--dimension")) {
      if (i + 1 < argc) {
        i++;
        p.wos.x0.dimension = atoi(argv[i]);
      } else {
        std::cerr << "--dimension option requires one argument." << std::endl;
        return 1;
      }
    } else if ((arg == "-it") || (arg == "--itterations")) {
      if (i + 1 < argc) {
        i++;
        p.wos.itterations = atoi(argv[i]);
      } else {
        std::cerr << "--itterations option requires one argument." << std::endl;
        return 1;
      }
    } else {
      // set default
      show_usage(argv);
      return 0;
    }
  }
}
