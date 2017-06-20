
/**
  * K nearest neighbor search using CUDA.
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param ind_host      indexes of the k nearest neighbors ; pointer to linear matrix
  */
void knnCudaWithIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);

/**
  * K nearest neighbor search using CUDA.
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distance to the k-th nearest neighbor for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k-th nearest neighbor ; pointer to linear matrix
  */
void knnCudaWithoutIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host);



/**
  * K nearest neighbor search using CUDA and CUBLAS.
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distance to the k-th nearest neighbor for each query point
  * - Copy distances from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k-th nearest neighbor ; pointer to linear matrix
  */
void knnCublasWithoutIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host);

/**
  * K nearest neighbor search using CUDA and CUBLAS.
  * - Initialize CUDA
  * - Allocate device memory
  * - Copy point sets (reference and query points) from host to device memory
  * - Compute the distances + indexes to the k nearest neighbors for each query point
  * - Copy distances and indexes from device to host memory
  *
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
  * @param ind_host      indexes of the k nearest neighbors ; pointer to linear matrix
  */
void knnCublasWithIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);