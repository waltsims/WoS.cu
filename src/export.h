#include "params.h"
#include <fstream>
#include "clock.h"

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

template <typename T>
void linearOutputData(const char *filename, T *h_exitX, T *h_exitY, int paths);

void exportData(double *h_paths, double *h_exitX, double *exitY, Parameters &p);

float getBase(int paths);

int getNumPoints(int paths);

int getCurrentPoint(int exponent, float base);

void exportData(double *h_paths, Parameters &p);

void exportTime(Timers &timers, Parameters &p);
