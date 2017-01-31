#include "params.h"

class Parameters; // forward declaration
// this is the cumsum devided by number of relative paths (insitu)
template <typename T>
void boundary2ExpectancyValue(T *vals, int paths);

// callculate the relative error (insitu)
template <typename T>
void getRelativeError(T *data, int paths, T end);

template <typename T>
void outputConvergence(const char *filename, T *vals, T *values, int paths);

void plot(double *h_paths, Parameters &p);
