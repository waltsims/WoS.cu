#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

enum SimulationTypes { nativeWos, thrustWos, hostWos };

size_t getSizeSharedMem(size_t len);

class X0 {
public:
  X0() : value(0.0), dimension(512) {}
  size_t dimension;

  float value;
};

class Parameters {
public:
  Parameters() : totalPaths(65535), eps(0.01), simulation(nativeWos) {}

  X0 x0;
  unsigned int numThreads;
  unsigned long int totalPaths;
  unsigned long int gpuPaths;
  int blockIterations;
  int blockRemainder;
  unsigned int numberBlocks;
  unsigned int nGPU;
  size_t size_SharedMemory;
  // TODO: Question: what effect does the d_eps have on practical convergence?
  float eps;
  SimulationTypes simulation;

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
