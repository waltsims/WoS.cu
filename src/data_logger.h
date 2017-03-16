#ifndef DATALOGGER_H
#define DATALOGGER_H
#include "gpu_config.h"
#include "parameters.h"
#include "timers.h"
#include <fstream>

class DataLogger {
public:
  DataLogger(Timers &timers, const Parameters p, const GPUConfig gpu)
      : timers(timers), p(p), gpu(gpu), simulationResult(0.0f), avgPath(0.0f),
        avgNumSteps(0.0f) {}
  void setAvgPathSet(float input) { avgPath = input; }
  void setAvgNumSteps(float input) { avgNumSteps = input; }
  void setResult(float result) { simulationResult = result; }
  float getResult() { return simulationResult; }
  float getNumSteps() { return avgNumSteps; }
  float getPath() { return avgPath; }

  void logData();

private:
  float simulationResult;
  float avgPath;
  float avgNumSteps;
  Timers &timers;
  const Parameters p;
  const GPUConfig gpu;

  off_t fsize(const char *filename);
  float calcRelativeError(float result, float exactSolution);
  void outputHeader(std::ofstream &file);
};

#endif // DATALOGGER_H
