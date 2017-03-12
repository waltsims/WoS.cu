#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include "export.h"
#include "helper.hpp"
#include "parse.h"
#include "timers.h"

#include "wos_wrapper.cuh"

// initialize h_x0 vector of size dim and fill with val

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

  // TODO: call WoS template wraper function

  Parameters p;

  // TODO this should/could go in parameter constructor
  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

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
