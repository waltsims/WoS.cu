
#include "clock.h"
#include "params.h"
#include <cstring>
#include <fstream>
#include <math.h>
#include <sys/stat.h>

// this is the cumsum devided by number of relative paths (insitu)
template <typename T>
void cumsum(T *vals, int paths) {
  for (int i = 1; i < paths; i++)
    vals[i] += vals[i - 1];
}

template <typename T>
void calcExpectedValue(T *vals, int paths) {
  cumsum(vals, paths);
  for (int i = 0; i < paths; i++) {
    vals[i] /= i + 1;
  }
}
// callculate the relative error (insitu)
template <typename T>
void calcRelativeError(T *data, int paths, T exactSolution) {
  for (int i = 0; i < paths; i++) {
    data[i] = fabs((data[i] - exactSolution) / exactSolution);
  }
}
void outputHeader(std::ofstream &file) {
  file << "paths,"
       << "relativeError,"
       << "expectedValue,"
       << "value" << std::endl;
}

float getBase(int paths, int points) { return pow(10, log(paths) / points); }

// int getNumPoints(int paths) {
//   return (int)ceil(log(paths / log(getBase(paths))));
// }

int getCurrentPoint(int exponent, float base) {
  return (int)ceil(pow(base, exponent));
}

template <typename T>
void logOutputData(const char *filename, T *relError, T *values,
                   T *expectedValue, int paths) {
  // TODO impliment for run numbers greater than MAX_BLOCKS

  printInfo("writing to file:");
  std::cout << "\t" << filename << std::endl;
  std::ofstream file(filename);
  outputHeader(file);
  // TODO remove logarithmic recution to seperate function
  int data_points = 50; // define number out ouput points desired
  int last = 0;         // to avoid printing the same point twice

  int exponent = 0;
  int currentPoint = getCurrentPoint(exponent, getBase(paths, data_points));

  while (currentPoint < paths) {
    if (last != currentPoint) {
      file << currentPoint << "," << relError[currentPoint] << ","
           << expectedValue[currentPoint] << "," << values[currentPoint]
           << std::endl;
      last = currentPoint;
    }
    exponent++;
    currentPoint = getCurrentPoint(exponent, getBase(paths, data_points));
  }
  file.close();
}

template <typename T>
void linearOutputData(const char *filename, T *h_exitX, T *h_exitY, int paths) {
  std::cout << "writing file to: " << filename << std::endl;

  std::ofstream file(filename);
  file << "exitX,"
       << "exitY" << std::endl;

  for (int currentPoint = 0; currentPoint < paths; currentPoint++) {
    file << h_exitX[currentPoint] << "," << h_exitY[currentPoint] << std::endl;
  }
  file.close();
}

template <typename T>
void linearOutputData(const char *filename, T *relError, T *values,
                      T *expectedValue, int paths) {
  std::cout << "writing file to: " << filename << std::endl;

  std::ofstream file(filename);
  outputHeader(file);
  for (int currentPoint = 0; currentPoint < paths; currentPoint++) {
    file << currentPoint << "," << relError[currentPoint] << ","
         << expectedValue[currentPoint] << "," << values[currentPoint]
         << std::endl;
  }
  file.close();
}

void exportData(double *h_paths, double *h_exitX, double *h_exitY,
                Parameters &p) {
  typedef double T;
  // T end = vals[paths - 1]; // use last value to test convergence
  T exactSolution = 0.29468541312605526226;

  T *data = (T *)malloc(p.wos.totalPaths * sizeof(double));
  T *expectedValue = (T *)malloc(p.wos.totalPaths * sizeof(double));
  std::memcpy(data, h_paths, p.wos.totalPaths * sizeof(double));

  // std::cout << "data\n" << std::endl;
  // for (int i = 0; i < p.wos.totalPaths; i++)
  //   std::cout << data[i] << std::endl;

  printInfo("exporting simulation data");
  calcExpectedValue(data, p.wos.totalPaths);
  std::memcpy(expectedValue, data, p.wos.totalPaths * sizeof(double));
  calcRelativeError(data, p.wos.totalPaths, exactSolution);
  logOutputData("docs/data/cuWos_data.csv", data, h_paths, expectedValue,
                p.wos.totalPaths);

  linearOutputData("docs/data/exit_positions.csv", h_exitX, h_exitY,
                   p.wos.totalPaths);

  free(data);
  free(expectedValue);
}

void exportData(double *h_paths, Parameters &p) {
  typedef double T;
  // T end = vals[paths - 1]; // use last value to test convergence
  T exactSolution = 0.29468541312605526226;

  T *data = (T *)malloc(p.wos.totalPaths * sizeof(double));
  T *expectedValue = (T *)malloc(p.wos.totalPaths * sizeof(double));
  std::memcpy(data, h_paths, p.wos.totalPaths * sizeof(double));

  printInfo("exporting simulation data");
  calcExpectedValue(data, p.wos.totalPaths);
  std::memcpy(expectedValue, data, p.wos.totalPaths * sizeof(double));
  calcRelativeError(data, p.wos.totalPaths, exactSolution);
  logOutputData("docs/data/cuWos_data.csv", data, h_paths, expectedValue,
                p.wos.totalPaths);

  free(data);
  free(expectedValue);
}
// source:
// http://stackoverflow.com/questions/8236/how-do-you-determine-the-size-of-a-file-in-c

off_t fsize(const char *filename) {
  struct stat st;

  if (stat(filename, &st) == 0)
    return st.st_size;

  return -1;
}

void exportTime(Timers &timers, Parameters &p) {
  // write time header

  std::ofstream file("docs/data/timing_data.csv",
                     std::ofstream::out | std::ofstream::app);

  if (fsize("docs/data/timing_data.csv") == 0) {
    file << "number of dimesions, number of threads, number of paths, "
            "number of blocks, computation time, total time, data "
            "initailization "
            "time, data download time"
         << std::endl;
  }
  file << p.wos.x0.dimension << "," << p.wos.x0.dimension << ","
       << p.wos.totalPaths << "," << p.wos.totalPaths << ","
       << timers.computationTimer.get() << "," << timers.totalTimer.get() << ","
       << timers.memorySetupTimer.get() << ","
       << timers.memoryDownloadTimer.get() << std::endl;
  file.close();
}