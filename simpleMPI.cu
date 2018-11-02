/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/* Simple example demonstrating how to use MPI with CUDA
*
*  Generate some random numbers on one node.
*  Dispatch them to all nodes.
*  Compute their square root on each node's GPU.
*  Compute the average of the results using MPI.
*
*  simpleMPI.cu: GPU part, compiled with nvcc
*/

#include <iostream>
using std::cerr;
using std::endl;

#include "simpleMPI.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
        my_abort(err); }


// Device code
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(float *input_a, float *input_b, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = sqrt(input_a[tid]*input_a[tid] + input_b[tid]*input_b[tid]);
}


// Initialize an array with random data (between 0 and 1)
void initData(float *data, int dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// CUDA computation on each node
// No MPI here, only CUDA
void computeGPU(float *hostData_a, float *hostData_b, int blockSize, int gridSize)
{
    int dataSize = blockSize * gridSize;

    // Allocate data on GPU memory
    float *deviceInputData_a = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceInputData_a, dataSize * sizeof(float)));
    float *deviceInputData_b = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceInputData_b, dataSize * sizeof(float)));

    float *deviceOutputData = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, dataSize * sizeof(float)));

    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(deviceInputData_a, hostData_a, dataSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInputData_b, hostData_b, dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    simpleMPIKernel<<<gridSize, blockSize>>>(deviceInputData_a, deviceInputData_b, deviceOutputData);

    // Copy data back to CPU memory
    CUDA_CHECK(cudaMemcpy(hostData_a, deviceOutputData, dataSize *sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(deviceInputData_a));
    CUDA_CHECK(cudaFree(deviceInputData_b));
    CUDA_CHECK(cudaFree(deviceOutputData));
}

float max_here(float *data, int size)
{
    float max_val = data[0];

    for (int i = 1; i < size; i++)
    {
        if(data[i] > max_val){
            max_val = data[i];
        }
    }

    return max_val;
}
