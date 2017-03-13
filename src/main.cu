#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include "export.h"
#include "helper.h"
#include "timers.h"

#include "wos_wrapper.cuh"

void printConfig(Parameters p, GPUConfig gpu);

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

  // initialize parameters object
  Parameters p = Parameters::parseParams(argc, argv);
  GPUConfig gpu = GPUConfig::createConfig(p);
  // TODO: simulationConfig class
  printConfig(p, gpu);

  // instantiate timers
  Timers timers;

  timers.totalTimer.start();

  // Calling WoS kernel
  float gpu_result = wosWrapper(timers, p, gpu);

  timers.totalTimer.end();

#ifdef OUT
  logData(gpu_result, timers, p, gpu);
#endif

  testResults(gpu_result, p);
  printTiming(timers.memorySetupTimer.get(), timers.computationTimer.get(),
              timers.totalTimer.get(), timers.memoryDownloadTimer.get());

  return (0);
}

// output params
void printConfig(Parameters p, GPUConfig gpu) {
  printf("CONFIGURATION:\n\tX0:\t\t\t%f\n\tWoS dimension:\t\t%zu\n\tWoS "
         "totalPaths:\t\t%ld\n\tnumber of blocks:\t%d\n"
         "\tIterations per blocks:\t%d\n\tremainder per "
         "blocks:\t%d\n\tnumThreads:\t\t%d\n\teps:\t\t\t%f\n\tlogging:\t\t%s\n",
         p.x0.value, p.x0.dimension, p.totalPaths, gpu.numberBlocks,
         gpu.blockIterations, gpu.blockRemainder, gpu.numThreads, p.eps,
         (p.logging) ? "true" : "false");
}
