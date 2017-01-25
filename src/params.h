#ifndef PARAMS_H
#define PARAMS_H

#include <iostream>
#include <stdlib.h>
#include <string>

#include "helper.hpp"

// TODO make class that holds both structure and function

template <typename T>
size_t getSizeSharedMem(size_t len) {
  // definition of total size needed for variable in buffer dependent on the
  // length of the data transefered
  return (4 * len) * sizeof(T);
}

class ReductionParameters {
public:
  int blocks;
  int threads;
};

class WoSParameters {
public:
  struct X0 {
    size_t dimension;
    size_t length;

    // TODO: should be a template param. hard to impiment

    double value;
  };
  X0 x0;
  int iterations;
};

class Parameters {
public:
  ReductionParameters reduction;
  WoSParameters wos;
};

#endif // PARAMS_H
