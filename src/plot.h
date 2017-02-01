#include "params.h"

class Parameters; // forward declaration
// this is the cumsum devided by number of relative paths (insitu)
template <typename T>
void calcExpectedValue(T *vals, int paths);

// callculate the relative error (insitu)
template <typename T>
void calcRelativeError(T *data, int paths, T end);

template <typename T>
void cumsum(T *vals, int paths);

void outputHeader(std::ofstream &file);

template <typename T>
void logOutputData(const char *filename, T *vals, T *values, int paths);

template <typename T>
void linearOutputData(const char *filename, T *vals, T *values, int paths);

void plot(double *h_paths, Parameters &p);
