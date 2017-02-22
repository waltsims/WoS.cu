#include <fstream>
#include "timers.h"

class Parameters; // forward declaration
// this is the cumsum devided by number of relative paths (insitu)
void calcExpectedValue(float *vals, int paths);

// callculate the relative error (insitu)
void calcRelativeError(float *data, int paths, float exactSolution);

void cumsum(float *vals, int paths);

void outputHeader(std::ofstream &file);

void logOutputData(const char *filename, float *vals, float *values, int paths);

void linearOutputData(const char *filename, float *vals, float *values,
                      int paths);

void linearOutputData(const char *filename, float *h_exitX, float *h_exitY,
                      int paths);

void exportData(float *h_paths, float *h_exitX, float *exitY, Parameters &p);

float getBase(int paths);

int getNumPoints(int paths);

int getCurrentPoint(int exponent, float base);

void exportData(float *h_paths, Parameters &p);

void exportTime(Timers &timers, Parameters &p);
