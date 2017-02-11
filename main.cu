#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include "inc/helper_cuda.h"
#include "src/clock.h"
#include "src/export.h"
#include "src/parse.h"

#include "src/wos_thrust.cuh"
#include "src/wos_kernel.cuh"

#include <limits>
#include <math_functions.h>

// initialize h_x0 vector of size dim and fill with val

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

// TODO: call WoS template wraper function

#ifdef DOUBLE
  typedef double T; // Type for problem
#else
  typedef float T;
#endif // DOUBLE
  Parameters p;

  // TODO this should/could go in parameter constructor
  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

  // instantiate timers
  Timers timers;

  timers.totalTimer.start();

  // Calling WoS kernel

  T gpu_result = wos<T>(timers, p);

  timers.totalTimer.end();

  testResults((float)gpu_result, p);
  printTiming(timers.memorySetupTimer.get(), timers.computationTimer.get(),
              timers.totalTimer.get(), timers.memoryDownloadTimer.get());
  exportTime(timers, p);

  return (0);
}
