#ifndef CLOCK_H
#define CLOCK_H

#include <ctime>
#include <cuda_runtime.h>

class Timer {
public:
  Timer() : tStart(0), running(false), sec(0.f) {}
  void start() {
    tStart = clock();
    running = true;
  }
  void end() {
    if (!running) {
      sec = 0;
      return;
    }
    cudaDeviceSynchronize();
    clock_t tEnd = clock();
    sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
    running = false;
  }
  float get() {
    if (running)
      end();
    return sec;
  }

private:
  clock_t tStart;
  bool running;
  float sec;
};

class Timers {
public:
  Timer computationTimer;
  Timer totalTimer;
  Timer memorySetupTimer;
  Timer memoryDownloadTimer;
};
#endif // CLOCK_H
