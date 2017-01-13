#include "helper.hpp"

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

// Finds the next largest power 2 number
extern "C" size_t nextPow2(size_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}
