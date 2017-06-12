#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

#include "cuda.h"
#include "cublas.h"


#include "knncuda.h"


/**
 * Initialize the random generator.
 */
void initRandomGenerator()
{
  struct timeval time; 
  gettimeofday(&time, NULL);
  srand(time.tv_usec * time.tv_sec);
}


/**
 * Initialize the 2D array that contains the points randomly.
 * @param array  2D array to initialize
 * @param nb     number of points
 * @param dim    dimension of the points
 */
void setRandomValues(float * array, const int nb, const int dim)
{
  for (int i=0; i<nb*dim; ++i) {
    array[i] = rand() / static_cast<float>(RAND_MAX);
  }
}


/**
  * Example of use of kNN search CUDA.
  */
int main() {
  
  // Parameters
  int ref_nb     = 1000;    // Reference point number, max=65535
  int query_nb   = 1000;     // Query point number,     max=65535
  int dim        = 32;      // Dimension of points,    max=8192
  int k          = 20;     // Nearest neighbors to consider
  int iterations = 10;    // Number of iterations for statistics
  
  // Allocation of points
  float * ref   = new float[ref_nb   * dim];
  float * query = new float[query_nb * dim];

  // Random initialization
  initRandomGenerator();
  setRandomValues(ref,   ref_nb,   dim);
  setRandomValues(query, query_nb, dim);

  // Allocation of distance and index arrays
  float * dist = new float[query_nb * k];
  int *   ind  = new int[query_nb * k];
  
  // Display informations
  printf("Reference points number : %6d\n", ref_nb);
  printf("Query points number     : %6d\n", query_nb);
  printf("Dimension of points     : %6d\n", dim);
  printf("Number of neighbors     : %6d\n", k);
  printf("Processing kNN search   :");

  // Timer variables
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time;
  
  // Call kNN search CUDA
  cudaEventRecord(start, 0);
  // for (i=0; i<iterations; i++)
  //   knn(ref, ref_nb, query, query_nb, dim, k, dist);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time/1000, iterations, elapsed_time/(iterations*1000));
  
  // Destroy cuda event object and free memory
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Clean-up memory
  delete[] ref;
  delete[] query;
  delete[] dist;
  delete[] ind;
}
