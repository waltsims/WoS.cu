#ifndef PARAMS_H
#define PARAMS_H

#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "helper.hpp"

// TODO make class that holds both structure and function

size_t getLength(size_t dim);

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads);

template <bool isDouble>
size_t getSizeSharedMem(size_t len);

unsigned int getRunsPerBlock(unsigned int runs, unsigned int &number_blocks);

class ReductionParameters {
public:
  int blocks;
  int threads;
};

class WoSParameters {
public:
  struct X0 {
    size_t dimension;
    size_t length;

    // TODO: should be a template param. hard to impiment

    double value;
  };

  X0 x0;
  int totalPaths;
  int pathsPerBlock;
  size_t size_SharedMemory;
  bool typeDouble; // 0 for float 1 for double
};

class Parameters {
public:
  ReductionParameters reduction;
  WoSParameters wos;
};

#endif // PARAMS_H
