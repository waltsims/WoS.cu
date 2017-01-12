#ifndef PARAMS_H
#define PARAMS

#include <iostream>
#include <stdlib.h>
#include <string>

class commandlineParams {
private:
  static void show_usage(char *argv[]) {
    std::cerr
        << "Usage: " << argv[0] << " <option(s)> SOURCES"
        << "Options:\n"
        << "\t-h,\t--help\tShow this help message\n"
        << "\t-nbr,\t--numBlocks NUMBLOCKS \t specify number of blocks for \n"
           "reduction"
        << "\t-ntr,\t--numThreads NUMTHREADS \t specify number of Threads for "
           "reduction"
        << std::endl;
  }

public:
  int reductionBlocks;
  int reductionThreads;

  int parseParams(int argc, char *argv[]) {
    // set default values
    commandlineParams clp;
    clp.reductionThreads = 512;
    clp.reductionBlocks = 256;

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
          std::cerr << "--numThreads option requires one argument."
                    << std::endl;
          return 1;
        }
      } else {
        // set default
        show_usage(argv);
      }
    }
  }
};

#endif // PARAMS_H
