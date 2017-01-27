
#include "params.h"
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
void getRelativeError(T *vals, int paths) {

  T end = vals[paths - 1];

  for (int i = 0; i < paths; i++) {
    vals[i] = fabs((vals[i] - end) / end);
  }
}

template <typename T>
void outputConvergence(const char *filename, T *vals, int paths) {
  // TODO impliment for run numbers greater than MAX_BLOCKS
  std::ofstream file(filename);
  file << "paths\t"
       << "solution val\t" << std::endl;
  // only export every 1.25^n val reduce file size
  // TODO remove logarithmic recution to seperate functionk
  int points = ceil(log(paths) / log(1.25));

  for (int i = 0; i < points; i++) {
    file << ceil(pow(1.25, i)) << "\t" << vals[i] / pow(1.25, i) << "\t"
         << std::endl;
  }
  file.close();
}
void plot(double *h_paths, Parameters &p) {

  printf("exporting convergences data\n");
  boundary2ExpectancyValue(h_paths, p.wos.totalPaths);
  getRelativeError(h_paths, p.wos.totalPaths);
  outputConvergence("docs/data/cuWos_convergence.dat", h_paths,
                    p.wos.totalPaths);
}
