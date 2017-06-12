__global__ void cuComputeDistanceTexture(texture<float, 2, cudaReadModeElementType> & texA, int wA, float * B, int wB, int pB, int dim, float* AB);

__global__ void cuComputeDistanceGlobal( float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB);

__global__ void cuComputeNorm(float *mat, int width, int pitch, int height, float *norm);

__global__ void cuAddRNorm(float *dist, int width, int pitch, int height, float *vec);

__global__ void cuAddQNormAndSqrt(float *vec1,  float *vec2, int width);

__global__ void cuInsertionSort(float *dist, int width, int pitch, int height, int k);

__global__ void cuInsertionSortWithIndexes(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k);

__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k);

void printErrorMessage(cudaError_t error, int memorySize);

