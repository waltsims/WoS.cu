#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// TODO make class that holds both structure and function

enum SimulationTypes { nativeWos, thrustWos, hostWos };

template <bool isDouble>
size_t getSizeSharedMem(size_t len);

class ReductionParameters {
public:
  int blocks;
  int threads;
};

class X0 {
public:
  X0() : value(0.0), dimension(512) {}
  size_t dimension;

  // TODO: should be a template param. hard to impliment

  float value;
};

class WoSParameters {
public:
  WoSParameters() : totalPaths(65535), eps(0.01), simulation(nativeWos) {}

  X0 x0;
  unsigned int numThreads;
  unsigned long int totalPaths;
  int blockIterations;
  int blockRemainder;
  unsigned int numberBlocks;
  size_t size_SharedMemory;
  // TODO: Question: what effect does the d_eps have on practical convergence?
  float eps;
  SimulationTypes simulation;
};

class Parameters {
public:
  ReductionParameters reduction;
  WoSParameters wos;

  void update() {
    updateNumBlocksAndThreads();
    updateNumThreads();
    updateSizeSharedMemory();
    updateEps();
  };

  void outputParameters(int count);

private:
  void updateEps();
  void updateNumBlocksAndThreads();
  void updateNumThreads();
  void updatePathsPerBlock();
  void updateSizeSharedMemory();
  unsigned int primeFactor();
};

#endif // PARAMS_H
