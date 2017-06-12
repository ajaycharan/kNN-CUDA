#include <stdio.h>

#include "cuda.h"
#include "cublas.h"

#include "knncuda_kernels.h"
#include "knncuda_constants.h"


// Define a cuda texture type for lisibility only
typedef texture<float, 2, cudaReadModeElementType> cudaTexture;


/**
 * Check the amount of available free memory in bytes.
 * @param memoryFree output free memory in bytes
 * @return true if everything went fine, false otherwise
 */
bool getFreeMemoryInBytes(size_t & memoryFree)
{
    // Get the first CUDA device
    CUdevice device;
    if (cuDeviceGet(&device, 0)!=CUDA_SUCCESS) {
        printf("Unable to get CUDA device");
        return false;
    }

    // Create the context
    CUcontext context;
    if (cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device)!=CUDA_SUCCESS) {
        printf("Unable to create the CUDA context");
        return false;
    }

    // Check the free and total memory values in bytes
    size_t memory_total;
    if (cuMemGetInfo(&memoryFree, &memory_total)!=CUDA_SUCCESS) {
        printf("Unable to access memory information");
        return false;
    }

    // Detactch the context
    if (cuCtxDetach(context)!=CUDA_SUCCESS) {
        printf("Unable to detach the context");
        return false;
    }

    return true;
}


/**
 * Can we use a texture to store the reference points?
 * @param refWidth  width (number) of reference points
 * @param refHeight height (dimension) of reference points
 * @return true if the texture can be used, false otherwise
 */
bool isTextureUsable(const int refWidth, const int refHeight)
{
    if (refWidth * sizeof(float) > MAX_TEXTURE_WIDTH_IN_BYTES)
        return false;
    else if (refHeight * sizeof(float) > MAX_TEXTURE_HEIGHT_IN_BYTES)
        return false;
    else
        return true;
}


void knnCudaWithoutIndexes(float* ref_host,
                           int    ref_width,
                           float* query_host,
                           int    query_width,
                           int    height,
                           int    k,
                           float* dist_host){
    
    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory in bytes
    size_t memory_free;
    if (!getFreeMemoryInBytes(memory_free)) {
        return;
    }
    
    // We might not be able to treat all query points at one. So we cut them in batchs.
    // We compute here the number of points we'll consier in each batch and the number
    // of batchs we'll have to do.
    size_t batch_size = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - ref_width * height * sizeof(float) ) / ((height + ref_width) * sizeof(float));
    batch_size = min((size_t)query_width, (batch_size / 16) * 16 );
    const int batch_nb = ceil(query_width / (float)batch_size);
    
    // Allocate memory for query points
    float  * query_dev;
    size_t   query_pitch_in_bytes;
    cudaError_t result = cudaMallocPitch( (void **)&query_dev, &query_pitch_in_bytes, batch_size * sizeof(float), height);
    if (result!=cudaSuccess) {
        printf("Error: Unable to allocate query memory\n");
        return;
    }
    const size_t query_pitch = query_pitch_in_bytes / sizeof(float);

    // Allocate memory for query points
    float  * dist_dev;
    size_t   tmp_pitch;
    result = cudaMallocPitch( (void **)&dist_dev, &tmp_pitch, batch_size * sizeof(float), ref_width);
    if (result!=cudaSuccess) {
        printf("Error: Unable to allocate distance memory\n");
        cudaFree(query_dev);
        return;
    }

    // Check if we can use the texture
    const bool use_texture = isTextureUsable(ref_width, height);

    // Allocation of memory (global or texture) for reference points
    cudaTexture    ref_texture;        // texture
    cudaArray    * ref_array;          // texture
    float        * ref_dev;            // global
    size_t         ref_pitch;
    if (use_texture){
    
        // Allocation of texture memory
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        result = cudaMallocArray(&ref_array, &channel_desc, ref_width, height);
        if (result!=cudaSuccess) {
            printf("Error: Unable to allocate reference point memory (texture)\n");
            cudaFree(query_dev);
            cudaFree(dist_dev);
            return;
        }

        // Copy the memory from host to device
        cudaMemcpyToArray(ref_array, 0, 0, ref_host, ref_width * height * sizeof(float), cudaMemcpyHostToDevice);
        
        // Set texture parameters and bind texture to the array
        ref_texture.addressMode[0] = cudaAddressModeClamp;
        ref_texture.addressMode[1] = cudaAddressModeClamp;
        ref_texture.filterMode     = cudaFilterModePoint;
        ref_texture.normalized     = 0;
        cudaBindTextureToArray(ref_texture, ref_array);
        
    }
    else{
    
        // Allocate memory in globlal memory
        size_t ref_pitch_in_bytes;
        result = cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes, ref_width * sizeof(float), height);
        if (result!=cudaSuccess) {
            printf("Error: Unable to allocate reference point memory (global)\n");
            cudaFree(query_dev);
            cudaFree(dist_dev);
            return;
        }
        ref_pitch = ref_pitch_in_bytes / sizeof(float);

        // Copy memory from host to device
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width * sizeof(float), ref_width * sizeof(float), height, cudaMemcpyHostToDevice);
    }
    
    // Split queries to fit in GPU memory
    for (int i=0; i<batch_nb; ++i){
        
        // Number of query points considered
        const int actual_query_width = min(query_width - i * batch_nb, batch_nb);
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i*batch_nb], query_width * sizeof(float), actual_query_width * sizeof(float), height, cudaMemcpyHostToDevice);
        
        // Grid for 16x16 threads
        dim3 t_16x16(16, 16, 1);
        dim3 g_16x16(actual_query_width/16, ref_width/16, 1);
        if (actual_query_width%16 != 0)
            g_16x16.x += 1;
        if (ref_width  %16 != 0)
            g_16x16.y += 1;

        // // Grid for 256x1 threads
        // dim3 t_256x1(256, 1, 1);
        // dim3 g_256x1(actual_query_width/256, 1, 1);
        // if (actual_query_width%256 != 0)
        //     g_256x1.x += 1;
        
        // Kernel 1: Compute all the distances
        if (use_texture)
            cuComputeDistanceTexture<<<g_16x16,t_16x16>>>(ref_texture, ref_width, query_dev, actual_query_width, query_pitch, height, dist_dev);
        else
            cuComputeDistanceGlobal<<<g_16x16,t_16x16>>>(ref_dev, ref_width, ref_pitch, query_dev, actual_query_width, query_pitch, height, dist_dev);
            
        // Kernel 2: Sort each column
        cuInsertionSort<<<g_256x1,t_256x1>>>(dist_dev, actual_query_width, query_pitch, ref_width, k);
        
        // Kernel 3: Compute square root of k-th element
        // cuParallelSqrt<<<g_256x1,t_256x1>>>(dist_dev+(k-1)*query_pitch, query_width);
        
        // Memory copy of output from device to host
        cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev+(k-1)*query_pitch, query_pitch_in_bytes, actual_nb_query_width*size_of_float, 1, cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    if (use_texture) {
        cudaFreeArray(ref_array);
    }
    else {
        cudaFree(ref_dev);
    }
    cudaFree(query_dev);
    cudaFree(dist_dev);
}




