
#include "params.h"
#include <cstring>
#include <fstream>
#include <math.h>

// this is the cumsum devided by number of relative paths (insitu)
template <typename T>
void boundary2ExpectancyValue(T *vals, int paths) {

  for (int i = 1; i < paths; i++) {
    vals[i] += vals[i - 1];
    vals[i - 1] /= i;
  }
  vals[paths - 1] /= paths;
}
// callculate the relative error (insitu)
template <typename T>
void getRelativeError(T *data, int paths, T end) {

  for (int i = 0; i < paths; i++) {
    data[i] = fabs((data[i] - end) / end);
  }
}

template <typename T>
void outputConvergence(const char *filename, T *data, T *values, int paths) {
  // TODO impliment for run numbers greater than MAX_BLOCKS
  std::cout << "writing file to: " << filename << std::endl;
  std::ofstream file(filename);
  file << "paths\t"
       << "relative error\t"
       << "value\t" << std::endl;
  // only export every 1.25^n val reduce file size
  // TODO remove logarithmic recution to seperate functionk
  int points = (int)ceil(log(paths) / log(1.25));

  for (int i = 2; i < points; i++) {
    file << (int)floor(pow(1.25, i)) << "\t" << data[(int)floor(pow(1.25, i))]
         << "\t" << values[(int)floor(pow(1.25, i))] << std::endl;
  }
  file.close();
}
void plot(double *h_paths, Parameters &p) {
  typedef double T;
  // T end = vals[paths - 1]; // use last value to test convergence
  T end = 0.29468541312605526226;

  T *data = (T *)malloc(p.wos.totalPaths * sizeof(double));
  std::memcpy(data, h_paths, p.wos.totalPaths * sizeof(double));

  // std::cout << "data\n" << std::endl;
  // for (int i = 0; i < p.wos.totalPaths; i++)
  //   std::cout << data[i] << std::endl;

  printf("exporting convergences data\n");
  boundary2ExpectancyValue(data, p.wos.totalPaths);
  getRelativeError(data, p.wos.totalPaths, end);
  outputConvergence("docs/data/cuWos_convergence.dat", data, h_paths,
                    p.wos.totalPaths);
  free(data);
}
