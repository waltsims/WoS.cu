//#include <cublas_v2.h>
#ifndef THRUST
#define THRUST

#include "params.h"
#include "timers.h"

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

// source:
// http://stackoverflow.com/questions/12614164/generating-a-random-number-vector-between-0-and-1-0-using-thrust

struct prg {
  float a, b;
  __host__ __device__ prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b){};

  __host__ __device__ float operator()(unsigned int thread_id) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<float> dist(a, b);
    rng.discard(thread_id);

    return dist(rng);
  }
};

struct getBoundaryDistanceThrust {
  float width;
  getBoundaryDistanceThrust(float _width) { width = _width; }

  __host__ __device__ float operator()(float &radius) const {
    return (1.0 - abs(radius));
  }
};

float wosThrust(Timers &timers, Parameters &p) {

  timers.memorySetupTimer.start();
  thrust::host_vector<float> h_x0(p.wos.x0.dimension);
  thrust::device_vector<float> d_x(p.wos.x0.dimension);
  thrust::device_vector<float> d_x0(p.wos.x0.dimension);
  thrust::fill_n(d_x0.begin(), p.wos.x0.dimension, p.wos.x0.value);
  thrust::device_vector<float> d_radius(p.wos.x0.dimension);
  thrust::fill(d_radius.begin(), d_radius.end(), INFINITY);
  thrust::device_vector<float> d_direction(p.wos.x0.dimension);
  thrust::fill(d_direction.begin(), d_direction.end(), 0.0);
  thrust::device_vector<float> d_paths(p.wos.totalPaths);
  thrust::fill(d_paths.begin(), d_paths.end(), 0.0);

#ifdef OUT
  thrust::device_vector<float> d_exitX(p.wos.totalPaths);
  thrust::fill(d_exitX.begin(), d_exitX.end(), 0.0);
  thrust::device_vector<float> d_exitY(p.wos.totalPaths);
  thrust::fill(d_exitY.begin(), d_exitY.end(), 0.0);
#endif // OUT

  timers.memorySetupTimer.end();
  timers.computationTimer.start();
  float radius = INFINITY;
  float norm = 0.0;
  unsigned int position;
  float gpu_result = 0;
  double d_eps = p.wos.eps;
  float sum = 0.0;
  float squaredSum = 0.0;
  unsigned int counter = 0;
  unsigned int randCount = 0;

  thrust::counting_iterator<unsigned int> index_sequence_begin(0);
  for (unsigned int i = 0; i < p.wos.totalPaths; i++) {
    thrust::copy(d_x0.begin(), d_x0.end(), d_x.begin());

    radius = INFINITY;
    norm = 0.0;
    position = 0;
    counter = 0;
    while (d_eps <= radius) {
      // std::cout << " direction before rand:" << std::endl;
      // thrust::copy(d_direction.begin(), d_direction.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;
      // create random direction
      thrust::transform(index_sequence_begin + randCount,
                        index_sequence_begin + p.wos.x0.dimension + randCount,
                        d_direction.begin(), prg(0.0, 1.0));
      // std::cout << " direction after rand:" << std::endl;
      // thrust::copy(d_direction.begin(), d_direction.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;

      sum += thrust::reduce(d_direction.begin(), d_direction.end());
      squaredSum += thrust::inner_product(
          d_direction.begin(), d_direction.end(), d_direction.begin(), 0.f);
      randCount += p.wos.x0.dimension;

      // normalize random direction
      // Source:
      // http://stackoverflow.com/questions/13688307/how-to-normalise-a-vector-with-thrust
      // std::cout << " direction before inner product:" << std::endl;
      // thrust::copy(d_direction.begin(), d_direction.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;
      norm = std::sqrt(thrust::inner_product(
          d_direction.begin(), d_direction.end(), d_direction.begin(), 0.f));
      // std::cout << " direction after inner_product:" << std::endl;
      // thrust::copy(d_direction.begin(), d_direction.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;

      using namespace thrust::placeholders;

      thrust::transform(d_direction.begin(), d_direction.end(),
                        d_direction.begin(), _1 / norm);

      thrust::transform(d_x.begin(), d_x.end(), d_radius.begin(),
                        getBoundaryDistanceThrust(1.f));

      // calculate mimimun radius
      // #if (CUDART_VERSION == 8000)
      //       // Source:
      //       //
      //       http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
      //       Thrust::device_vector<float>::iterator iter =
      //           thrust::min_element(d_radius.begin(), d_radius.end());
      // #elif (CUDART_VERSION == 7050)
      thrust::detail::normal_iterator<thrust::device_ptr<float>> iter =
          thrust::min_element(d_radius.begin(), d_radius.end());
      // #endif
      radius = *iter;

      // std::cout << " before step:" << std::endl;
      // thrust::copy(d_x.begin(), d_x.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;

      // calculate next point X
      thrust::transform(d_direction.begin(), d_direction.end(), d_x.begin(),
                        d_x.end(), _2 += radius * _1);

      // std::cout << " after step:" << std::endl;
      // thrust::copy(d_x.begin(), d_x.end(),
      //              std::ostream_iterator<float>(std::cout, " "));
      // std::cout << "\n" << std::endl;

      counter++;
    }
    // std::cout << "while itterations: " << counter << std::endl;
    // Project current point to boundary
    thrust::transform(d_x.begin(), d_x.end(), d_radius.begin(),
                      getBoundaryDistanceThrust(1.f));

    // #if (CUDART_VERSION == 8000)
    //     // Source:
    //     //
    //     http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
    //     Thrust::device_vector<float>::iterator iter =
    //         thrust::min_element(d_radius.begin(), d_radius.end());
    // #elif (CUDART_VERSION == 7050)
    thrust::detail::normal_iterator<thrust::device_ptr<float>> iter =
        thrust::min_element(d_radius.begin(), d_radius.end());
    // #endif

    position = iter - d_radius.begin();
    radius = *iter;

    // std::cout << " before projection:" << std::endl;
    // thrust::copy(d_x.begin(), d_x.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;

    // TODO: keep sign
    // project closest dimension to boundary
    thrust::fill(d_x.begin() + position, d_x.begin() + position + 1, 1.f);
#ifdef OUT
    if (p.wos.x0.dimension == 2) {
      d_exitX[i] = d_x[0];
      d_exitY[i] = d_x[1];
    }
#endif
    // std::cout << "before boundary eval:" << std::endl;
    // thrust::copy(d_x.begin(), d_x.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;

    // evaluate boundary value
    d_paths[i] =
        thrust::inner_product(d_x.begin(), d_x.end(), d_x.begin(), 0.f) /
        (p.wos.x0.dimension * 2);

    // std::cout << "result vector in iteration " << i << " : " << std::endl;
    // thrust::copy(d_paths.begin(), d_paths.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;
  }
  std::cout << "mean: " << sum / randCount << std::endl;
  std::cout << "standard deviation: "
            << sqrt(pow(sum / randCount, 2) + squaredSum / randCount)
            << std::endl;
  timers.computationTimer.end();
#ifdef OUT
  thrust::host_vector<float> h_paths(p.wos.totalPaths);
  thrust::host_vector<float> h_exitPoints(p.wos.x0.dimension *
                                          p.wos.totalPaths);
  thrust::host_vector<float> h_exitX(p.wos.totalPaths);
  thrust::host_vector<float> h_exitY(p.wos.totalPaths);
  thrust::copy(d_paths.begin(), d_paths.end(), h_paths.begin());
  thrust::copy(d_exitX.begin(), d_exitX.end(), h_exitX.begin());
  thrust::copy(d_exitY.begin(), d_exitY.end(), h_exitY.begin());

  exportData(h_paths.data(), h_exitX.data(), h_exitY.data(), p);

#endif // OUT
#if (CUDART_VERSION == 7050)
  gpu_result = thrust::reduce(d_paths.begin(), d_paths.end());
#elif (CUDART_VERSION == 8000)
  gpu_result = thrust::reduce(thrust::device, d_paths.begin(), d_paths.end());
#endif
  gpu_result /= p.wos.totalPaths;
  timers.totalTimer.end();

  return gpu_result;
}

#endif // !THRUST
