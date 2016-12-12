// source: cuda reduction documentation

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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// reduce6 from cuda documentation
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void sumReduce(T *g_idata, T *g_odata, unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  __syncthreads();

#if (__CUDA_ARCH__ >= 300)
  if (tid < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64)
      mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      mySum += __shfl_down(mySum, offset);
    }
  }
#else
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) {
    sdata[tid] = mySum = mySum + sdata[tid + 32];
  }

  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata[tid] = mySum = mySum + sdata[tid + 16];
  }

  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata[tid] = mySum = mySum + sdata[tid + 8];
  }

  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata[tid] = mySum = mySum + sdata[tid + 4];
  }

  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata[tid] = mySum = mySum + sdata[tid + 2];
  }

  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata[tid] = mySum = mySum + sdata[tid + 1];
  }

  __syncthreads();
#endif

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = mySum;
}

template <typename T>
void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata) {

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  if (isPow2(size)) {
    switch (threads) {
    case 512:
      sumReduce<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 256:
      sumReduce<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 128:
      sumReduce<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 64:
      sumReduce<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 32:
      sumReduce<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 16:
      sumReduce<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 8:
      sumReduce<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 4:
      sumReduce<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 2:
      sumReduce<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;

    case 1:
      sumReduce<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                             size);
      break;
    }
  } else {
    switch (threads) {
    case 512:
      sumReduce<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 256:
      sumReduce<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 128:
      sumReduce<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata,
                                                                d_odata, size);
      break;

    case 64:
      sumReduce<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 32:
      sumReduce<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 16:
      sumReduce<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                               size);
      break;

    case 8:
      sumReduce<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 4:
      sumReduce<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 2:
      sumReduce<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;

    case 1:
      sumReduce<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata,
                                                              size);
      break;
    }
  }
}

template void reduce<float>(int size, int threads, int blocks, float *d_idata,
                            float *d_odata);

template void reduce<double>(int size, int threads, int blocks, double *d_idata,
                             double *d_odata);
