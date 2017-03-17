#include "data_logger.h"
#include "helper.h"
#include "timers.h"

#include <iostream>
#include <math.h>
#include <sys/stat.h>

// source:
// http://stackoverflow.com/questions/8236/how-do-you-determine-the-size-of-a-file-in-c
off_t DataLogger::fsize(const char *filename) {
  struct stat st;
  if (stat(filename, &st) == 0)
    return st.st_size;

  return -1;
}

// callculate the relative error
float DataLogger::calcRelativeError(float result, float exactSolution) {
  return fabs(result - exactSolution) / exactSolution;
}

void DataLogger::outputHeader(std::ofstream &file) {
  file << "nDimensions,nThreads,nPaths,result,relErrror,exacltSolution,eps,"
          "avgPath,avgNumSteps,compTime,initTime,finishTime,totalTime"
       << std::endl;
}

void DataLogger::logData() {
  // write time header
  const char *filename = "docs/data/wos_log.csv";
  float exactSolution = (p.x0.dimension == 2) ? 0.29468541312605526226 : 0.0;
  float relError = calcRelativeError(simulationResult, exactSolution);

  std::ofstream file(filename, std::ofstream::out | std::ofstream::app);

  if (p.verbose) {
    std::cout << "writing log to: " << filename << std::endl;
  }
  if (fsize(filename) == 0) {
    outputHeader(file);
  }

  file << p.x0.dimension << "," << gpu.numThreads << "," << p.totalPaths << ","
       << simulationResult << "," << relError << "," << exactSolution << ","
       << p.eps << "," << getPath() << "," << getNumSteps() << ","
       << timers.computationTimer.get() << "," << timers.memorySetupTimer.get()
       << "," << timers.memoryDownloadTimer.get() << ","
       << timers.totalTimer.get() << std::endl;
  file.close();
}
