void knnCudaWithIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host){
    
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);
    
    // Variables
    float        *query_dev;
    float        *ref_dev;
    float        *dist_dev;
    int          *ind_dev;
    cudaArray    *ref_array;
    cudaError_t  result;
    size_t       query_pitch;
    size_t       query_pitch_in_bytes;
    size_t       ref_pitch;
    size_t       ref_pitch_in_bytes;
    size_t       ind_pitch;
    size_t       ind_pitch_in_bytes;
    size_t       max_nb_query_traited;
    size_t       actual_nb_query_width;
    size_t memory_total;
    size_t memory_free;
    
    
    // Check if we can use texture memory for reference points
    unsigned int use_texture = ( ref_width*size_of_float<=MAX_TEXTURE_WIDTH_IN_BYTES && height*size_of_float<=MAX_TEXTURE_HEIGHT_IN_BYTES );
    
    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);
    
    // Determine maximum number of query that can be treated
    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*height ) / ( size_of_float * (height + ref_width) + size_of_int * k);
    max_nb_query_traited = min((size_t)query_width, (max_nb_query_traited / 16) * 16 );
    
    // Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width);
    if (result){
        printErrorMessage(result, max_nb_query_traited*size_of_float*(height+ref_width));
        return;
    }
    query_pitch = query_pitch_in_bytes/size_of_float;
    dist_dev    = query_dev + height * query_pitch;
    
    // Allocation of global memory for indexes  
    result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
    if (result){
        cudaFree(query_dev);
        printErrorMessage(result, max_nb_query_traited*size_of_int*k);
        return;
    }
    ind_pitch = ind_pitch_in_bytes/size_of_int;
    
    // Allocation of memory (global or texture) for reference points
    if (use_texture){
    
        // Allocation of texture memory
        cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();
        result = cudaMallocArray( &ref_array, &channelDescA, ref_width, height );
        if (result){
            printErrorMessage(result, ref_width*height*size_of_float);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        cudaMemcpyToArray( ref_array, 0, 0, ref_host, ref_width * height * size_of_float, cudaMemcpyHostToDevice );
        
        // Set texture parameters and bind texture to array
        texA.addressMode[0] = cudaAddressModeClamp;
        texA.addressMode[1] = cudaAddressModeClamp;
        texA.filterMode     = cudaFilterModePoint;
        texA.normalized     = 0;
        cudaBindTextureToArray(texA, ref_array);
        
    }
    else{
    
        // Allocation of global memory
        result = cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height);
        if (result){
            printErrorMessage(result,  ref_width*size_of_float*height);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        ref_pitch = ref_pitch_in_bytes/size_of_float;
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float,  ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    }
    
    // Split queries to fit in GPU memory
    for (int i=0; i<query_width; i+=max_nb_query_traited){
        
        // Number of query points considered
        actual_nb_query_width = min( max_nb_query_traited, (size_t)(query_width-i));
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);
        
        // Grids ans threads
        dim3 g_16x16(actual_nb_query_width/16, ref_width/16, 1);
        dim3 t_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_16x16.x += 1;
        if (ref_width  %16 != 0) g_16x16.y += 1;
        //
        dim3 g_256x1(actual_nb_query_width/256, 1, 1);
        dim3 t_256x1(256, 1, 1);
        if (actual_nb_query_width%256 != 0) g_256x1.x += 1;
        //
        dim3 g_k_16x16(actual_nb_query_width/16, k/16, 1);
        dim3 t_k_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_k_16x16.x += 1;
        if (k  %16 != 0) g_k_16x16.y += 1;
        
        // Kernel 1: Compute all the distances
        if (use_texture)
            cuComputeDistanceTexture<<<g_16x16,t_16x16>>>(ref_width, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
        else
            cuComputeDistanceGlobal<<<g_16x16,t_16x16>>>(ref_dev, ref_width, ref_pitch, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
            
        // Kernel 2: Sort each column
        cuInsertionSortWithIndexes<<<g_256x1,t_256x1>>>(dist_dev, query_pitch, ind_dev, ind_pitch, actual_nb_query_width, ref_width, k);
        
        // Kernel 3: Compute square root of k first elements
        cuParallelSqrt<<<g_k_16x16,t_k_16x16>>>(dist_dev, query_width, query_pitch, k);
        
        // Memory copy of output from device to host
        cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev, query_pitch_in_bytes, actual_nb_query_width*size_of_float, k, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(&ind_host[i],  query_width*size_of_int,   ind_dev,  ind_pitch_in_bytes,   actual_nb_query_width*size_of_int,   k, cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    if (use_texture)
        cudaFreeArray(ref_array);
    else
        cudaFree(ref_dev);
    cudaFree(ind_dev);
    cudaFree(query_dev);
}


void knnCublasWithoutIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host){
    
    unsigned int size_of_float = sizeof(float);
    
    // Variables
    float        *dist_dev;
    float        *query_dev;
    float        *ref_dev;
    float        *query_norm;
    float        *ref_norm;
    size_t       query_pitch;
    size_t       query_pitch_in_bytes;
    size_t       ref_pitch;
    size_t       ref_pitch_in_bytes;
    size_t       max_nb_query_traited;
    size_t       actual_nb_query_width;
    size_t memory_total;
    size_t memory_free;
    cudaError_t  result;
    
    // CUDA Initialisation
    cuInit(0);
    cublasInit();
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);
    
    // Determine maximum number of query that can be treated
    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width * (height+1) ) / ( size_of_float * (height + ref_width + 1) );
    max_nb_query_traited = min((size_t)query_width, (max_nb_query_traited / 16) * 16 );
    
    // Allocation of global memory for query points, ||query||, and for 2.R^T.Q
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, (height + ref_width + 1));
    if (result){
        printErrorMessage(result, max_nb_query_traited * size_of_float * ( height + ref_width + 1 ) );
        return;
    }
    query_pitch = query_pitch_in_bytes/size_of_float;
    query_norm  = query_dev  + height * query_pitch;
    dist_dev    = query_norm + query_pitch;
    
    // Allocation of global memory for reference points and ||query||
    result = cudaMallocPitch((void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height+1);
    if (result){
        printErrorMessage(result, ref_width * size_of_float * ( height+1 ));
        cudaFree(query_dev);
        return;
    }
    ref_pitch = ref_pitch_in_bytes / size_of_float;
    ref_norm  = ref_dev + height * ref_pitch;
    
    // Memory copy of ref_host in ref_dev
    result = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float, ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    
    // Computation of reference square norm
    dim3 G_ref_norm(ref_width/256, 1, 1);
    dim3 T_ref_norm(256, 1, 1);
    if (ref_width%256 != 0) G_ref_norm.x += 1;
    cuComputeNorm<<<G_ref_norm,T_ref_norm>>>(ref_dev, ref_width, ref_pitch, height, ref_norm);
    
    // Main loop: split queries to fit in GPU memory
    for (int i=0;i<query_width;i+=max_nb_query_traited){
        
        // Nomber of query points actually used
        actual_nb_query_width = min(max_nb_query_traited, (size_t)(query_width-i));
        
        // Memory copy of ref_host in ref_dev
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);
        
        // Computation of Q square norm
        dim3 G_query_norm(actual_nb_query_width/256, 1, 1);
        dim3 T_query_norm(256, 1, 1);
        if (actual_nb_query_width%256 != 0) G_query_norm.x += 1;
        cuComputeNorm<<<G_query_norm,T_query_norm>>>(query_dev, actual_nb_query_width, query_pitch, height, query_norm);
        
        // Computation of Q*transpose(R)
        cublasSgemm('n', 't', (int)query_pitch, (int)ref_pitch, height, (float)-2.0, query_dev, query_pitch, ref_dev, ref_pitch, (float)0.0, dist_dev, query_pitch);
        
        // Add R norm to distances
        dim3 grid(actual_nb_query_width/16, ref_width/16, 1);
        dim3 thread(16, 16, 1);
        if (actual_nb_query_width%16 != 0) grid.x += 1;
        if (ref_width%16 != 0) grid.y += 1;
        cuAddRNorm<<<grid,thread>>>(dist_dev, actual_nb_query_width, query_pitch, ref_width,ref_norm);
        
        // Sort each column
        cuInsertionSort<<<G_query_norm,T_query_norm>>>(dist_dev,actual_nb_query_width,query_pitch,ref_width,k);
        
        // Add Q norm and compute Sqrt ONLY ON ROW K-1
        cuAddQNormAndSqrt<<<G_query_norm,T_query_norm>>>( dist_dev+(k-1)*query_pitch, query_norm, actual_nb_query_width);
        
        // Memory copy
        cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev+(k-1)*query_pitch, query_pitch_in_bytes, actual_nb_query_width*size_of_float, 1, cudaMemcpyDeviceToHost);
        
    }
    
    // Free memory
    cudaFree(ref_dev);
    cudaFree(query_dev);
    
    // CUBLAS shutdown
    cublasShutdown();
}



void knnCublasWithIndexes(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host){
    
    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);
    
    // Variables
    float        *query_dev;
    float        *ref_dev;
    float        *dist_dev;
    int          *ind_dev;
    cudaArray    *ref_array;
    cudaError_t  result;
    size_t       query_pitch;
    size_t       query_pitch_in_bytes;
    size_t       ref_pitch;
    size_t       ref_pitch_in_bytes;
    size_t       ind_pitch;
    size_t       ind_pitch_in_bytes;
    size_t       max_nb_query_traited;
    size_t       actual_nb_query_width;
    size_t memory_total;
    size_t memory_free;
    
    
    // Check if we can use texture memory for reference points
    unsigned int use_texture = ( ref_width*size_of_float<=MAX_TEXTURE_WIDTH_IN_BYTES && height*size_of_float<=MAX_TEXTURE_HEIGHT_IN_BYTES );
    
    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    cuCtxDetach (cuContext);
    
    // Determine maximum number of query that can be treated
    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*height ) / ( size_of_float * (height + ref_width) + size_of_int * k);
    max_nb_query_traited = min((size_t)query_width, (max_nb_query_traited / 16) * 16 );
    
    // Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width);
    if (result){
        printErrorMessage(result, max_nb_query_traited*size_of_float*(height+ref_width));
        return;
    }
    query_pitch = query_pitch_in_bytes/size_of_float;
    dist_dev    = query_dev + height * query_pitch;
    
    // Allocation of global memory for indexes  
    result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
    if (result){
        cudaFree(query_dev);
        printErrorMessage(result, max_nb_query_traited*size_of_int*k);
        return;
    }
    ind_pitch = ind_pitch_in_bytes/size_of_int;
    
    // Allocation of memory (global or texture) for reference points
    if (use_texture){
    
        // Allocation of texture memory
        cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();
        result = cudaMallocArray( &ref_array, &channelDescA, ref_width, height );
        if (result){
            printErrorMessage(result, ref_width*height*size_of_float);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        cudaMemcpyToArray( ref_array, 0, 0, ref_host, ref_width * height * size_of_float, cudaMemcpyHostToDevice );
        
        // Set texture parameters and bind texture to array
        texA.addressMode[0] = cudaAddressModeClamp;
        texA.addressMode[1] = cudaAddressModeClamp;
        texA.filterMode     = cudaFilterModePoint;
        texA.normalized     = 0;
        cudaBindTextureToArray(texA, ref_array);
        
    }
    else{
    
        // Allocation of global memory
        result = cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height);
        if (result){
            printErrorMessage(result,  ref_width*size_of_float*height);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        ref_pitch = ref_pitch_in_bytes/size_of_float;
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float,  ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    }
    
    // Split queries to fit in GPU memory
    for (int i=0; i<query_width; i+=max_nb_query_traited){
        
        // Number of query points considered
        actual_nb_query_width = min( max_nb_query_traited, (size_t)(query_width-i) );
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, actual_nb_query_width*size_of_float, height, cudaMemcpyHostToDevice);
        
        // Grids ans threads
        dim3 g_16x16(actual_nb_query_width/16, ref_width/16, 1);
        dim3 t_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_16x16.x += 1;
        if (ref_width  %16 != 0) g_16x16.y += 1;
        //
        dim3 g_256x1(actual_nb_query_width/256, 1, 1);
        dim3 t_256x1(256, 1, 1);
        if (actual_nb_query_width%256 != 0) g_256x1.x += 1;
        //
        dim3 g_k_16x16(actual_nb_query_width/16, k/16, 1);
        dim3 t_k_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_k_16x16.x += 1;
        if (k  %16 != 0) g_k_16x16.y += 1;
        
        // Kernel 1: Compute all the distances
        if (use_texture)
            cuComputeDistanceTexture<<<g_16x16,t_16x16>>>(ref_width, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
        else
            cuComputeDistanceGlobal<<<g_16x16,t_16x16>>>(ref_dev, ref_width, ref_pitch, query_dev, actual_nb_query_width, query_pitch, height, dist_dev);
            
        // Kernel 2: Sort each column
        cuInsertionSortWithIndexes<<<g_256x1,t_256x1>>>(dist_dev, query_pitch, ind_dev, ind_pitch, actual_nb_query_width, ref_width, k);
        
        // Kernel 3: Compute square root of k first elements
        cuParallelSqrt<<<g_k_16x16,t_k_16x16>>>(dist_dev, query_width, query_pitch, k);
        
        // Memory copy of output from device to host
        cudaMemcpy2D(&dist_host[i], query_width*size_of_float, dist_dev, query_pitch_in_bytes, actual_nb_query_width*size_of_float, k, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(&ind_host[i],  query_width*size_of_int,   ind_dev,  ind_pitch_in_bytes,   actual_nb_query_width*size_of_int,   k, cudaMemcpyDeviceToHost);
    }
    
    // Free memory
    if (use_texture)
        cudaFreeArray(ref_array);
    else
        cudaFree(ref_dev);
    cudaFree(ind_dev);
    cudaFree(query_dev);
}