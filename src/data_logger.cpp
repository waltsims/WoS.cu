#include "data_logger.h"
#include "helper.h"
#include "timers.h"

#include <cstring>
#include <fstream>
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
  return fabs((result - exactSolution) / exactSolution);
}

void DataLogger::outputHeader(std::ofstream &file) {
  file << "number of dimensions, number of Threads, number of "
          "paths, result, relative errror, exacltSolution, eps,"
          "computation time,total time, data "
          "initailization time, data download time"
       << std::endl;
}

void DataLogger::logData() {
  // write time header
  const char *filename = "docs/data/wos_log.csv";
  float exactSolution = 0.29468541312605526226;
  float relError = calcRelativeError(simulationResult, exactSolution);

  std::ofstream file(filename, std::ofstream::out | std::ofstream::app);

  std::cout << "writing log to: " << filename << std::endl;
  if (fsize(filename) == 0) {
    outputHeader(file);
  }

  file << p.x0.dimension << "," << gpu.numThreads << "," << p.totalPaths << ","
       << simulationResult << "," << relError << "," << exactSolution << ","
       << p.eps << "," << timers.computationTimer.get() << ","
       << timers.totalTimer.get() << "," << timers.memorySetupTimer.get() << ","
       << timers.memoryDownloadTimer.get() << std::endl;
  file.close();
}
