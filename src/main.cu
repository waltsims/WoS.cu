#ifndef MAX_THREADS
#define MAX_THREADS 1024
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 65535
#endif

#include "../inc/helper_cuda.h"
#include "clock.h"
#include "parse.h"
#include "plot.h"

#ifndef THRUST
#include "wos_kernel.cuh"
#endif

#include <limits>
#include <math_functions.h>

//#include <cublas_v2.h>
#ifdef THRUST
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

template <typename T>
struct prg {
  T a, b;
  __host__ __device__ prg(T _a = 0.f, T _b = 1.f) : a(_a), b(_b){};

  __host__ __device__ T operator()(unsigned int thread_id) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<T> dist(a, b);
    rng.discard(thread_id);

    return dist(rng);
  }
};

template <typename T>
struct getBoundaryDistance {
  T width;
  getBoundaryDistance(T _width) { width = _width; }

  __host__ __device__ T operator()(T &radius) const {
    return (1 - abs(radius));
  }
};
#endif

// initialize h_x0 vector of size dim and fill with val
template <typename T>
void initX0(T *x0, size_t dim, size_t len, T val);

int main(int argc, char *argv[]) {
  printTitle();
  printInfo("initializing");

// TODO: call WoS template wraper function

#ifdef DOUBLE
  typedef double T; // Type for problem
#else
  typedef float T;
#endif // DOUBLE
  Parameters p;

  // TODO this should/could go in parameter constructor
  int parseStatus = parseParams(argc, argv, p);
  if (parseStatus == 0)
    return 0;

  // TODO: Question: what effect does the d_eps have on practical convergence?
  T d_eps = 0.01; // 1 / sqrt(p.wos.x0.dimension); // or 0.01

  // instantiate timers
  Timers timers;

  timers.totalTimer.start();
#ifndef THRUST
  // declare local array variabls
  T *h_x0;
  checkCudaErrors(cudaMallocHost((void **)&h_x0, sizeof(T) * p.wos.x0.length));
  // declare pointers for device variables
  T *d_x0 = NULL;
  T *d_paths = NULL;
  // init our point on host
  // cast to T hotfix until class is templated
  initX0(h_x0, p.wos.x0.dimension, p.wos.x0.length, (T)p.wos.x0.value);

  timers.memorySetupTimer.start();

  // maloc device memory
  checkCudaErrors(cudaMalloc((void **)&d_x0, p.wos.x0.length * sizeof(T)));

  printInfo("initializing d_paths");

  checkCudaErrors(cudaMalloc((void **)&d_paths, p.wos.totalPaths * sizeof(T)));

  checkCudaErrors(cudaMemset(d_paths, 0.0, p.wos.totalPaths * sizeof(T)));

  // Let's bring our data to the Device
  checkCudaErrors(cudaMemcpy(d_x0, h_x0, p.wos.x0.length * sizeof(T),
                             cudaMemcpyHostToDevice));

  timers.memorySetupTimer.end();
  timers.computationTimer.start();

  // Calling WoS kernel
  wos<T>(p, d_x0, d_paths, d_eps);
  cudaDeviceSynchronize();
  timers.computationTimer.end();

  // We don't need d_x0 anymore, only to reduce solution data
  cudaFree(d_x0);

#endif

#ifdef THRUST
  timers.memorySetupTimer.start();
  thrust::host_vector<T> h_x0(p.wos.x0.dimension);
  thrust::device_vector<T> d_x(p.wos.x0.dimension);
  thrust::device_vector<T> d_x0(p.wos.x0.dimension);
  thrust::fill_n(d_x0.begin(), p.wos.x0.dimension, (T)p.wos.x0.value);
  thrust::device_vector<T> d_radius(p.wos.x0.dimension);
  thrust::fill(d_radius.begin(), d_radius.end(), INFINITY);
  thrust::device_vector<T> d_direction(p.wos.x0.dimension);
  thrust::fill(d_direction.begin(), d_direction.end(), 0.0);
  thrust::device_vector<T> d_paths(p.wos.totalPaths);
  thrust::fill(d_paths.begin(), d_paths.end(), 0.0);

  timers.memorySetupTimer.end();
  timers.computationTimer.start();
  T radius = INFINITY;
  T norm = 0.0;
  unsigned int position;
  T gpu_result = 0;

  thrust::counting_iterator<T> index_sequence_begin(0);

  for (unsigned int i = 0; i < p.wos.totalPaths; i++) {
    thrust::copy(d_x0.begin(), d_x0.end(), d_x.begin());

    radius = INFINITY;
    norm = 0.0;
    position = 0;
    while (d_eps <= radius) {
      // create random direction
      thrust::transform(index_sequence_begin + p.wos.x0.dimension * i,
                        index_sequence_begin + p.wos.x0.dimension * (i + 1),
                        d_direction.begin(), prg<T>(0.0, 1.0));

      // normalize random direction
      // Source:
      // http://stackoverflow.com/questions/13688307/how-to-normalise-a-vector-with-thrust
      norm = std::sqrt(thrust::inner_product(
          d_direction.begin(), d_direction.end(), d_direction.begin(), (T)0.0));

      using namespace thrust::placeholders;

      thrust::transform(d_direction.begin(), d_direction.end(),
                        d_direction.begin(), _1 / norm);

      thrust::transform(d_x.begin(), d_x.end(), d_radius.begin(),
                        getBoundaryDistance<T>((T)1.0));

      // calculate mimimun radius
      // Source:
      // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
      thrust::device_vector<T>::iterator iter =
          thrust::min_element(d_radius.begin(), d_radius.end());

      radius = *iter;

      // calculate next point X
      thrust::transform(d_direction.begin(), d_direction.end(), d_x.begin(),
                        d_x.end(), _2 += radius * _1);
    }

    // Project current point to boundary
    thrust::transform(d_x.begin(), d_x.end(), d_radius.begin(),
                      getBoundaryDistance<T>((T)1.0));

    // find min element in radius
    thrust::device_vector<T>::iterator iter =
        thrust::min_element(d_radius.begin(), d_radius.end());

    position = iter - d_radius.begin();
    radius = *iter;
    // project closest dimension to boundary
    thrust::fill(d_x.begin() + position, d_x.begin() + position + 1, (T)1.0);

    // std::cout << " before inner product" << std::endl;
    // thrust::copy(d_x.begin(), d_x.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;

    // evaluate boundary value
    d_paths[i] =
        thrust::inner_product(d_x.begin(), d_x.end(), d_x.begin(), (T)0.0);

    // std::cout << "result vector in iteration " << i << " : " << std::endl;
    // thrust::copy(d_paths.begin(), d_paths.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;
    d_paths[i] /= p.wos.x0.dimension * 2;
    // std::cout << "result vector in iteration " << i << " : " << std::endl;
    // thrust::copy(d_paths.begin(), d_paths.end(),
    //              std::ostream_iterator<float>(std::cout, " "));
    // std::cout << "\n" << std::endl;
  }
  timers.computationTimer.end();
#ifdef PLOT
  thrust::host_vector<T> h_paths(p.wos.totalPaths);
  thrust::copy(d_paths.begin(), d_paths.end(), h_paths.begin());
  plot(h_paths.data(), p);

#endif
  gpu_result = thrust::reduce(thrust::device, d_paths.begin(), d_paths.end());
  gpu_result /= p.wos.totalPaths;
  timers.totalTimer.end();

#endif
#ifndef THRUST
#if defined(PLOT) || defined(CPU_REDUCE)
  printInfo("downloading path data\n");
  timers.memoryDownloadTimer.start();

  T h_paths[p.wos.totalPaths];
  // Download paths data
  checkCudaErrors(cudaMemcpyAsync(
      &h_paths, d_paths, p.wos.totalPaths * sizeof(T), cudaMemcpyDeviceToHost));

  timers.memoryDownloadTimer.end();
#endif

#ifdef PLOT
  plot(h_paths, p);
#endif

#ifdef CPU_REDUCE

  T gpu_result = reduceCPU(h_paths, p.wos.totalPaths);

#else

  T *h_results = (T *)malloc(p.reduction.blocks * sizeof(T));

  T *d_results;
  cudaCheckErrors(
      cudaMalloc((void **)&d_results, p.reduction.blocks * sizeof(T)));

  cudaError err;
  reduce(p.wos.totalPaths, p.reduction.threads, p.reduction.blocks, d_paths,
         d_results);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("Reduction Kernel returned an error:\n %s\n",
           cudaGetErrorString(err));
  }

  timers.memoryDownloadTimer.start();

#ifdef DEBUG
  printf("[MAIN]: results values before copy:\n");
  for (int n = 0; n < p.reduction.blocks; n++) {
    printf("%f\n", h_results[n]);
  }
#endif

  // copy result from device to hostcudaStat =
  cudaCheckErrors(cudaMemcpy(h_results, d_results,
                             p.reduction.blocks * sizeof(T),
                             cudaMemcpyDeviceToHost));

  timers.memoryDownloadTimer.end();

  T gpu_result = 0.0;
  for (int i = 0; i < p.reduction.blocks; i++) {
    printf("iteration %d, %f\n", i, h_results[i]);
    gpu_result += h_results[i];
  }
  free(h_results);
#endif

  gpu_result /= p.wos.totalPaths;

#ifdef DEBUG
  printf("[MAIN]: results values after copy:\n");
  for (int n = 0; n < p.reduction.blocks; n++) {
    printf("%f\n", h_results[n]);
  }
#endif

  timers.totalTimer.end();

  testResults((float)h_x0[0], (float)d_eps, (float)gpu_result, p);

  cudaFreeHost(h_x0);
#endif // !THRUST

#ifdef THRUST
  testResults((float)h_x0[0], (float)d_eps, gpu_result, p);
#endif // THRUST

  printTiming(timers.memorySetupTimer.get(), timers.computationTimer.get(),
              timers.totalTimer.get(), timers.memoryDownloadTimer.get());

#ifndef CPU_REDUCE
  cudaFree(d_results);
#endif

  return (0);
}

template <typename T>
void initX0(T *h_x0, size_t dim, size_t len, T val) {
  // init our point on host
  for (unsigned int i = 0; i < dim; i++)
    // h_x0[i] = i == 1 ? 0.22 : 0;
    h_x0[i] = val;
  for (unsigned int i = dim; i < len; i++)
    h_x0[i] = 0.0;
}
