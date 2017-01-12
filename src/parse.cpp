
#include "params.h"

#include <iostream>
#include <stdlib.h>
#include <string>

// Source: http://www.cplusplus.com/articles/DEN36Up4/

static void show_usage(char *argv[]) {
  std::cerr << "Usage: " << argv[0] << " <option(s)> SOURCES"
            << "Options:\n"
            << "\t-h,--help\t\tShow this help message\n"
            << "\t-nbr,--numBlocks NUMBLOCKS \t specify number of blocks for "
               "reduction"
            << "\t-ntr,--numThreads NUMTHREADS \t specify number of Threads "
               "for reduction"
            << std::endl;
}

int parseParams(int argc, char *argv[], commandlineParams) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      show_usage(argv);
      return 0;
    } else if ((arg == "-nbr") || (arg == "--numBlocks")) {
      if (i + 1 < argc) {
        clp.reductionBlocks = atoi(argv[i++]);
      } else {
        std::cerr << "--numBlocks option requires one argument." << std::endl;
        return 1;
      }
    } else if ((arg == "-ntr") || (arg == "--numThreads")) {
      if (i + 1 < argc) {
        clp.reductionThreads = atoi(argv[i++]);
      } else {
        std::cerr << "--numThreads option requires one argument." << std::endl;
        return 1;
      }
    } else {
      // set default
      clp.reductionThreads = 512;
      clp.reductionBlocks = 256;
    }
  }
}
