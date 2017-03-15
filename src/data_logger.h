#ifndef DATALOGGER_H
#define DATALOGGER_H
#include "gpu_config.h"
#include "parameters.h"
#include "timers.h"
#include <fstream>

class DataLogger {
public:
  DataLogger(Timers &timers, const Parameters p, const GPUConfig gpu)
      : timers(timers), p(p), gpu(gpu) {}
  void setAvgPathSet(float avgPath) { avgPath = avgPath; }
  void setAvgNumSteps(float avgNumSteps) { avgNumSteps = avgNumSteps; }
  void setResult(float result) { result = result; }

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
