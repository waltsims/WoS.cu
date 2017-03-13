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
  size_t dimension;
  float value;
};

class Parameters {
public:
  X0 x0;
  const unsigned long int totalPaths;
  const SimulationTypes simulation;
  const float eps; // eps could change with successive itterations

  // all device dependent, only const with two same Graphics cards
  const unsigned long int gpuPaths;
  const int blockIterations;
  const unsigned int numThreads;
  const int blockRemainder;
  const unsigned int numberBlocks;
  const int nGPU;
  const size_t size_SharedMemory;

  static Parameters parseParams(int argc, char *argv[]);

private:
  Parameters(const unsigned int numThreads, const unsigned long int totalPaths,
             const unsigned long int gpuPaths, const int blockIterations,
             const int blockRemainder, const unsigned int numberBlocks,
             const int nGPU, const size_t size_SharedMemory, const float eps,
             const SimulationTypes simulation, const size_t x0Dimension,
             const float x0Value)
      : numThreads(numThreads), totalPaths(totalPaths), gpuPaths(gpuPaths),
        blockIterations(blockIterations), blockRemainder(blockRemainder),
        numberBlocks(numberBlocks), nGPU(nGPU),
        size_SharedMemory(size_SharedMemory), eps(eps), simulation(simulation) {
    x0.dimension = x0Dimension;
    x0.value = x0Value;
  }
};
#endif // PARAMS_H
