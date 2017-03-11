#ifndef CPU_REDUCE
#define CPU_REDUCE
#include <algorithm>
#include <cmath>
#include <vector>

#include <iostream>

// return true if a has a smaller absolute value, else false
bool smallerAbs(const float &a, const float &b) {
  return (std::abs(a) > std::abs(b));
}

float heapReduce(float *data, int size) {
  bool odd = (bool)(size % 2);
  int stride = (odd) ? (size - 1) / 2 : size / 2;
  float sum = 0;
  std::vector<float> v(data, data + size);
  std::make_heap(v.begin(), v.end(), smallerAbs);

  while (v.size() > 1) {
    std::pop_heap(v.begin(), v.end(), smallerAbs);
    sum = v.back();
    v.pop_back();

    std::pop_heap(v.begin(), v.end(), smallerAbs);
    sum += v.back();
    v.pop_back();
    v.push_back(sum);

    push_heap(v.begin(), v.end(), smallerAbs);
  }

  return v.back();
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
float reduceCPU(float *data, int size) {
  float sum = data[0];
  float c = 0.f;

  // TODO declare outside of for loop!
  for (int i = 1; i < size; i++) {
    float y = data[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}
#endif // CPU_REDUCE
