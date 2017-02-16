#ifndef PARAMS_H
#define PARAMS_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// TODO make class that holds both structure and function

enum SimulationTypes { nativeWos, thrustWos };

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
  size_t length;

  // TODO: should be a template param. hard to impliment

  double value;
};

class WoSParameters {
public:
  WoSParameters() : totalPaths(65535), eps(0.01), simulation(nativeWos) {}

  X0 x0;
  unsigned int totalPaths;
  unsigned int pathsPerBlock;
  size_t size_SharedMemory;
  // TODO: Question: what effect does the d_eps have on practical convergence?
  double eps;
  SimulationTypes simulation;
};

class Parameters {
public:
  ReductionParameters reduction;
  WoSParameters wos;

  void update() {
    updateNumBlocksAndThreads();
    updateLength();
    updateSizeSharedMemory();
  };

  void outputParameters(int count);

private:
  void updateNumBlocksAndThreads();
  void updateLength();
  void updatePathsPerBlock();
  void updateSizeSharedMemory();
};

#endif // PARAMS_H
