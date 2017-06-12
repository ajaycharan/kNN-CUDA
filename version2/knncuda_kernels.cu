
#include "cuda.h"
#include "cublas.h"

#include "knncuda_constants.h"


#pragma mark -
#pragma mark CUDA based functions


/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  * The matrix A is a texture.
  *
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param pB    pitch of matrix B given in number of columns
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceTexture(texture<float, 2, cudaReadModeElementType> & texA, int wA, float * B, int wB, int pB, int dim, float* AB){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ( xIndex<wB && yIndex<wA ){
        float ssd = 0;
        for (int i=0; i<dim; i++){
            float tmp  = tex2D(texA, (float)yIndex, (float)i) - B[ i * pB + xIndex ];
            ssd += tmp * tmp;
        }
        AB[yIndex * pB + xIndex] = ssd;
    }
}


/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param pA    pitch of matrix A given in number of columns
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param pB    pitch of matrix B given in number of columns
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
__global__ void cuComputeDistanceGlobal(float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB){

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];
    
    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Other variables
    float tmp;
    float ssd = 0;
    
    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * pA;
    step_B  = BLOCK_DIM * pB;
    end_A   = begin_A + (dim-1) * pA;
    
    // Conditions
    int cond0 = (begin_A + tx < wA); // used to write in shared memory
    int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
    int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix
    
    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
        
        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/pA + ty < dim){
            shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0;
        }
        else{
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1){
            for (int k = 0; k < BLOCK_DIM; ++k){
                tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }
        
        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1)
        AB[ (begin_A + ty) * pB + begin_B + tx ] = ssd;
}


#pragma mark -
#pragma mark CUBLAS based functions


/**
 * Given a matrix of size width*height, compute the square norm of each column.
 *
 * @param mat    : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param height : the number of rowm for a colum major storage matrix
 * @param norm   : the vector containing the norm of the matrix
 */
__global__ void cuComputeNorm(float *mat, int width, int pitch, int height, float *norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        float val, sum=0;
        int i;
        for (i=0;i<height;i++){
            val  = mat[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}


/**
 * Given the distance matrix of size width*height, adds the column vector
 * of size 1*height to each column of the matrix.
 *
 * @param dist   : the matrix
 * @param width  : the number of columns for a colum major storage matrix
 * @param pitch  : the pitch in number of column
 * @param height : the number of rowm for a colum major storage matrix
 * @param vec    : the vector to be added
 */
__global__ void cuAddRNorm(float *dist, int width, int pitch, int height, float *vec){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty]=vec[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        dist[yIndex*pitch+xIndex]+=shared_vec[ty];
}



/**
 * Given two row vectors with width column, adds the two vectors and compute
 * the square root of the sum. The result is stored in the first vector.
 *
 * @param vec1  : the first vector
 * @param vec2  : the second vector
 * @param width : the number of columns for a colum major storage matrix
 */
__global__ void cuAddQNormAndSqrt(float *vec1,  float *vec2, int width){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        vec1[xIndex] = sqrt(vec1[xIndex]+vec2[xIndex]);
    }
}


#pragma mark -
#pragma mark Common functions


/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist     distance matrix
  * @param width    width of the distance matrix
  * @param pitch    pitch of the distance matrix given in number of columns
  * @param height   height of the distance matrix
  * @param k        number of smallest distance to consider
  */
__global__ void cuInsertionSort(float *dist, int width, int pitch, int height, int k){

    // Variables
    int l,i,j;
    float *p;
    float v, max_value;
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (xIndex<width){
        
        // Pointer shift and max value
        p         = dist+xIndex;
        max_value = *p;
        
        // Part 1 : sort kth firt element
        for (l=pitch;l<k*pitch;l+=pitch){
            v = *(p+l);
            if (v<max_value){
                i=0; while (i<l && *(p+i)<=v) i+=pitch;
                for (j=l;j>i;j-=pitch)
                    *(p+j) = *(p+j-pitch);
                *(p+i) = v;
            }
            max_value = *(p+l);
        }
        
        // Part 2 : insert element in the k-th first lines
        for (l=k*pitch;l<height*pitch;l+=pitch){
            v = *(p+l);
            if (v<max_value){
                i=0; while (i<k*pitch && *(p+i)<=v) i+=pitch;
                for (j=(k-1)*pitch;j>i;j-=pitch)
                    *(p+j) = *(p+j-pitch);
                *(p+i) = v;
                max_value  = *(p+(k-1)*pitch);
            }
        }
    }
}



/**
  * Gathers k-th smallest distances for each column of the distance matrix in the top.
  *
  * @param dist        distance matrix
  * @param dist_pitch  pitch of the distance matrix given in number of columns
  * @param ind         index matrix
  * @param ind_pitch   pitch of the index matrix given in number of columns
  * @param width       width of the distance matrix and of the index matrix
  * @param height      height of the distance matrix and of the index matrix
  * @param k           number of neighbors to consider
  */
__global__ void cuInsertionSortWithIndexes(float *dist, int dist_pitch, int *ind, int ind_pitch, int width, int height, int k){

    // Variables
    int l, i, j;
    float *p_dist;
    int   *p_ind;
    float curr_dist, max_dist;
    int   curr_row,  max_row;
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (xIndex<width){
        
        // Pointer shift, initialization, and max value
        p_dist   = dist + xIndex;
        p_ind    = ind  + xIndex;
        max_dist = p_dist[0];
        p_ind[0] = 1;
        
        // Part 1 : sort kth firt elementZ
        for (l=1; l<k; l++){
            curr_row  = l * dist_pitch;
            curr_dist = p_dist[curr_row];
            if (curr_dist<max_dist){
                i=l-1;
                for (int a=0; a<l-1; a++){
                    if (p_dist[a*dist_pitch]>curr_dist){
                        i=a;
                        break;
                    }
                }
                for (j=l; j>i; j--){
                    p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
                    p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
                }
                p_dist[i*dist_pitch] = curr_dist;
                p_ind[i*ind_pitch]   = l+1;
            }
            else
                p_ind[l*ind_pitch] = l+1;
            max_dist = p_dist[curr_row];
        }
        
        // Part 2 : insert element in the k-th first lines
        max_row = (k-1)*dist_pitch;
        for (l=k; l<height; l++){
            curr_dist = p_dist[l*dist_pitch];
            if (curr_dist<max_dist){
                i=k-1;
                for (int a=0; a<k-1; a++){
                    if (p_dist[a*dist_pitch]>curr_dist){
                        i=a;
                        break;
                    }
                }
                for (j=k-1; j>i; j--){
                    p_dist[j*dist_pitch] = p_dist[(j-1)*dist_pitch];
                    p_ind[j*ind_pitch]   = p_ind[(j-1)*ind_pitch];
                }
                p_dist[i*dist_pitch] = curr_dist;
                p_ind[i*ind_pitch]   = l+1;
                max_dist             = p_dist[max_row];
            }
        }
    }
}


/**
  * Computes the square root of the first line (width-th first element)
  * of the distance matrix.
  *
  * @param dist    distance matrix
  * @param width   width of the distance matrix
  * @param pitch   pitch of the distance matrix given in number of columns
  * @param k       number of neighbors to consider
  */
__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}

