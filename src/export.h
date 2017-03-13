#include "GPUConfig.h"
#include "params.h"
#include "timers.h"
#include <fstream>

off_t fsize(const char *filename);

void calcExpectedValue(float *vals, int paths);

float calcRelativeError(float result, float exactSolution);

void outputHeader(std::ofstream &file);

void logData(float result, Timers &timers, Parameters &p, GPUConfig &gpu);
