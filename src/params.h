#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <stdlib.h>
#include <string>

class ReductionParameters {
public:
  int blocks;
  int threads;
};

class Parameters {
public:
  ReductionParameters reduction;
};

#endif // PARAMS_H
