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

float reduceCPU(float *data, int size) {
  bool odd = (bool)(size % 2);
  int stride = (odd) ? (size - 1) / 2 : size / 2;
  float sum = 0;
  std::vector<float> v(data, data + size);
  std::make_heap(v.begin(), v.end(), smallerAbs);

  // for (auto i : v) {
  //   std::cout << i << std::endl;
  // }

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
  // for (auto i : v) {
  //   std::cout << i << std::endl;
  // }

  return v.back();
}
#endif // CPU_REDUCE
