#ifndef PARAMS_H
#define PARAMS_H

#include "helper.h"
#include <stdio.h>

enum SimulationTypes { nativeWos, thrustWos, hostWos };

size_t getSizeSharedMem(size_t len);

class X0 {
public:
  size_t dimension; // should be constant even though value can change
  float value;
};

class Parameters {
public:
  X0 x0;
  const unsigned long long int totalPaths;
  const SimulationTypes simulation;
  const float eps; // eps could change with successive itterations
  const bool logging;
  const bool avgPath;
  const bool verbose;

  static Parameters parseParams(int argc, char *argv[]);

private:
  Parameters(const unsigned long long int totalPaths, const float eps,
             const SimulationTypes simulation, const size_t x0Dimension,
             const float x0Value, const bool logging, const bool avgPath,
             const bool verbose)
      : totalPaths(totalPaths), eps(eps), simulation(simulation),
        logging(logging), avgPath(avgPath), verbose(verbose) {
    x0.dimension = x0Dimension;
    x0.value = x0Value;
  }
};
#endif // PARAMS_H
