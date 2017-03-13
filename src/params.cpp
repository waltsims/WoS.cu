#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper.h"
#include "params.h"

// Source: http://www.cplusplus.com/articles/DEN36Up4/

static void show_usage(char *argv[]) {
  // TODO: update usage
  std::cerr
      << "Usage: " << argv[0] << " <option(s)> SOURCES"
      << "Options:\n"
      << "\t-h,\t--help\t\t\t\tShow this help message\n"
      << "\t-x0,\t--x0Value\t\t\tdefine consant value for x0\n"
      << "\t-dim,\t--dimension\t\t\tdefine the dimension of the problem. "
         "(ergo length of vector x0)\n"
      << "\t-it,\t--totalPaths\t\t\tdefine the number of iterations for the "
         "algorithm.\n"
      << "\t-eps,\t--eps\t\t\tdefine epsilon value for the algorithm.\n"
      << "\t-st,\t--simulation-type\t\t[thrust | native | host]\n"
      << std::endl;
}

Parameters Parameters::parseParams(int argc, char *argv[]) {
  unsigned int count = 0;
  // set default values
  // TODO: types are needed here
  unsigned long int totalPaths = 65535;
  float eps = 0.01;
  SimulationTypes simulation = nativeWos;
  size_t x0Dimension = 512;
  float x0Value = 0.0;
  int nGPU = 1;
  unsigned int numThreads;
  size_t size_SharedMemory;
  int blockRemainder;
  unsigned int numberBlocks;
  int blockIterations;
  unsigned long int gpuPaths;

  // TODO use exceptions
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if ((arg == "-h") || (arg == "--help")) {
        throw(std::runtime_error("please try again"));
      } else if ((arg == "-x0") || (arg == "--x0Value")) {
        if (i + 1 < argc) {
          i++;
          count++;
          x0Value = atof(argv[i]);
        } else {
          throw(std::runtime_error("--x0Value option requires one argument."));
        }
      } else if ((arg == "-dim") || (arg == "--dimension")) {
        if (i + 1 < argc) {
          i++;
          count++;
          x0Dimension = atoi(argv[i]);
        } else {
          throw(
              std::runtime_error("--dimension option requires one argument."));
        }
      } else if ((arg == "-it") || (arg == "--totalPaths")) {
        if (i + 1 < argc) {
          i++;
          count++;
          totalPaths = atoi(argv[i]);
        } else {
          throw(std::runtime_error(
              "--itterations option requires one argument."));
        }
      } else if ((arg == "-eps") || (arg == "--eps")) {
        if (i + 1 < argc) {
          i++;
          count++;
          eps = atof(argv[i]);
        } else {
          throw(std::runtime_error("--eps option requires one argument."));
        }
      } else if ((arg == "-st") || (arg == "--simulation-type")) {
        if (i + 1 < argc) {
          i++;
          count++;
          if (!strcmp(argv[i], "thrust"))
            simulation = thrustWos;
          else if (!strcmp(argv[i], "native"))
            simulation = nativeWos;
          else if (!strcmp(argv[i], "host"))
            simulation = hostWos;
          else {
            throw(std::runtime_error("--simulation-type option requires one "
                                     "argument.[thrust |host| native]"));
          }
        } else {
          throw(std::runtime_error("please try again"));
        }
      }
    } catch (...) {
      show_usage(argv);
      std::terminate();
    }
  }

  // updaate numThreads
  numThreads = (isPow2(x0Dimension)) ? x0Dimension : nextPow2(x0Dimension);
  // update size of shared memory
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  size_SharedMemory = (4 * numThreads + 1) * sizeof(float);
  // source::https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
  // uppdate paths per block
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0); // assume one device for now

  if (totalPaths <= MAX_BLOCKS) {
    numberBlocks = totalPaths;
    blockIterations = 1;
    blockRemainder = numberBlocks;
  } else {
    numberBlocks = MAX_BLOCKS;
    blockIterations = ceil((totalPaths / (float)MAX_BLOCKS));
    blockRemainder = totalPaths % MAX_BLOCKS;
  }

  // output params
  printf("Running Simulation with %d arguments\n", count);
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%ld\n\tnumber of blocks:\t%d\n"
         "\tIterations per blocks:\t%d\n\tremainder per "
         "blocks:\t%d\n\tnumThreads:\t\t%d\n\teps:\t\t\t%f\n",
         x0Value, x0Dimension, totalPaths, numberBlocks, blockIterations,
         blockRemainder, numThreads, eps);

  return Parameters(numThreads, totalPaths, gpuPaths, blockIterations,
                    blockRemainder, numberBlocks, nGPU, size_SharedMemory, eps,
                    simulation, x0Dimension, x0Value);
}
