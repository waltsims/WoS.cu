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

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

  // initialize parameters object
  Parameters p = Parameters::parseParams(argc, argv);

  // instantiate timers
  Timers timers;

  timers.totalTimer.start();

  // Calling WoS kernel
  float gpu_result = wosWrapper(timers, p);

  timers.totalTimer.end();

#ifdef OUT
  logData(gpu_result, timers, p);
#endif

  testResults(gpu_result, p);
  printTiming(timers.memorySetupTimer.get(), timers.computationTimer.get(),
              timers.totalTimer.get(), timers.memoryDownloadTimer.get());

  return (0);
}
