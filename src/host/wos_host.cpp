#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "../parameters.h"
#include "timers.h"

float wosHost(Timers &timers, Parameters &p) {
  unsigned int dimension = 2;
  unsigned int totalPaths = 100000;
  // declare local array variabls
  // init our point on host
  timers.memorySetupTimer.start();
  float *h_x0 = (float *)malloc(p.x0.dimension * sizeof(float));
  float *h_results = (float *)malloc(p.totalPaths * sizeof(float));
  float *h_x = (float *)malloc(p.x0.dimension * sizeof(float));
  float *radius = (float *)malloc(p.x0.dimension * sizeof(float));
  float *h_direction = (float *)malloc(p.x0.dimension * sizeof(float));

  // init x0
  for (unsigned int i = 0; i < p.x0.dimension; i++) {
    h_x0[i] = p.x0.value;
  }

  float r;
  float norm;
  unsigned int minIndex;
  float host_result = 0.0;
  float temp;
  timers.memorySetupTimer.end();

  timers.computationTimer.start();
  printf("setting up problem\n");
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 1);

  for (unsigned int currentPath = 0; currentPath < p.totalPaths;
       currentPath++) {

    std::memcpy((void *)h_x, (void *)h_x0, p.x0.dimension * sizeof(float));
    r = INFINITY;

    while (p.eps < r) {

      // printf("path: %d\n", currentPath);
      // std::memcpy((void *)cache, (void *)h_x, dimension * sizeof(float));

      for (unsigned int i = 0; i < p.x0.dimension; i++) {
        radius[i] = 1 - fabs(h_x[i]);
        r = (radius[i] < r) ? radius[i] : r;
        h_direction[i] = distribution(generator);
        norm += h_direction[i] * h_direction[i];
      }

      norm = sqrt(norm);

      for (unsigned int i = 0; i < p.x0.dimension; i++) {
        h_x[i] += r * h_direction[i] / norm;
      }
      // for (unsigned int i = 0; i < dimension; i++) {
      //   printf("%f ", h_x[i]);
      // }
      // printf("\n");
    }

    // printf("before projection: ");
    // for (unsigned int i = 0; i < diension; i++) {
    //   printf("%f ", h_x[i]);
    // }
    // printf("\n");

    minIndex = 0;
    for (unsigned int i = 0; i < p.x0.dimension; i++) {
      radius[i] = 1 - fabs(h_x[i]);
      minIndex = (radius[i] < radius[minIndex]) ? i : minIndex;
    }

    h_x[minIndex] = 1;
    // printf("after projection: ");
    // for (unsigned int i = 0; i < dimension; i++) {
    //   printf("%f ", h_x[i]);
    // }
    // printf("\n");

    temp = 0.0;
    for (unsigned int i = 0; i < p.x0.dimension; i++) {
      temp += h_x[i] * h_x[i];
    }
    host_result += temp / (2 * p.x0.dimension);
    // printf("result on iteration %d: %f\n", currentPath, temp / (2 *
    // dimension));
  }

  host_result /= (p.totalPaths);
  timers.computationTimer.end();

  timers.memoryDownloadTimer.start();
  free(h_x0), free(h_results), free(h_x), free(radius), free(h_direction);
  timers.memoryDownloadTimer.end();
  return host_result;
}
