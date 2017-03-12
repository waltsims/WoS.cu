
#include "helper.h"
#include "timers.h"
#include <cstring>
#include <fstream>
#include <math.h>
#include <sys/stat.h>

// source:
// http://stackoverflow.com/questions/8236/how-do-you-determine-the-size-of-a-file-in-c
off_t fsize(const char *filename) {
  struct stat st;
  if (stat(filename, &st) == 0)
    return st.st_size;

  return -1;
}

// callculate the relative error
float calcRelativeError(float result, float exactSolution) {
  return fabs((result - exactSolution) / exactSolution);
}

void outputHeader(std::ofstream &file) {
  file << "number of dimensions, number of Threads, number of "
          "paths, result, relative errror, exacltSolution, eps,"
          "computation time,total time, data "
          "initailization time, data download time"
       << std::endl;
}

void logData(float result, Timers &timers, Parameters &p) {
  // write time header
  const char *filename = "docs/data/wos_log.csv";
  float exactSolution = 0.29468541312605526226;
  float relError = calcRelativeError(result, exactSolution);

  std::ofstream file(filename, std::ofstream::out | std::ofstream::app);

  std::cout << "writing log to: " << filename << std::endl;
  if (fsize(filename) == 0) {
    outputHeader(file);
  }

  file << p.wos.x0.dimension << "," << p.wos.numThreads << ","
       << p.wos.totalPaths << "," << result << "," << relError << ","
       << exactSolution << "," << p.wos.eps << ","
       << timers.computationTimer.get() << "," << timers.totalTimer.get() << ","
       << timers.memorySetupTimer.get() << ","
       << timers.memoryDownloadTimer.get() << std::endl;
  file.close();
}
