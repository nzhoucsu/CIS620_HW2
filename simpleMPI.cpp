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
*  simpleMPI.cpp: main program, compiled with mpicxx on linux/Mac platforms
*                 on Windows, please download the Microsoft HPC Pack SDK 2008
*/

// MPI include
#include <mpi.h>
#include <unistd.h>

// System includes
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

// User include
#include "simpleMPI.h"

// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        cerr << "MPI error calling \""#call"\"\n"; \
        my_abort(-1); }


// Host code
// No CUDA here, only MPI
int main(int argc, char *argv[])
{
    // Dimensions of the dataset
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;

    // hostname
    char hostname[256];
    // mpi processor
    int myrank, nprocs;

    // Initialize MPI state
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Get our MPI node number and node count
    int commSize, commRank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    // Generate some random numbers on the root node (node 0)
    int dataSizeTotal = dataSizePerNode * commSize;
    float *dataRoot_A = NULL;
    float *dataRoot_B = NULL;

    if (commRank == 0)  // Are we the root node?
    {
        cout << "Running on " << commSize << " nodes" << endl;
        dataRoot_A = new float[dataSizeTotal];
        initData(dataRoot_A, dataSizeTotal);
        dataRoot_B = new float[dataSizeTotal];
        initData(dataRoot_B, dataSizeTotal);
    }

    // Allocate a buffer on each node
    float *dataNode_A = new float[dataSizePerNode];
    float *dataNode_B = new float[dataSizePerNode];

    // Dispatch a portion of the input data to each node
    MPI_CHECK(MPI_Scatter(dataRoot_A,
                          dataSizePerNode,
                          MPI_FLOAT,
                          dataNode_A,
                          dataSizePerNode,
                          MPI_FLOAT,
                          0,
                          MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatter(dataRoot_B,
                          dataSizePerNode,
                          MPI_FLOAT,
                          dataNode_B,
                          dataSizePerNode,
                          MPI_FLOAT,
                          0,
                          MPI_COMM_WORLD));

    if (commRank == 0)
    {
        // No need for root data any more
        delete [] dataRoot_A;
        delete [] dataRoot_B;
    }

    // On each node, run computation on GPU
    computeGPU(dataNode_A, dataNode_B, blockSize, gridSize);

    // Reduction to the root node, computing the sum of output elements
    float maxNode = max_here(dataNode_A, dataSizePerNode);
    gethostname(hostname, 256);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cout << "Processor " << myrank << " of " << nprocs << "\t" << hostname << "\t\toutput is " << maxNode << endl;

    float maxRoot;
    MPI_CHECK(MPI_Reduce(&maxNode, &maxRoot, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));

    if (commRank == 0)
    {
        float max_val = maxRoot;
        cout << "Maximum Euclidean distance is: " << max_val << endl;
    }

    // Cleanup
    delete [] dataNode_A;
    delete [] dataNode_B;
    MPI_CHECK(MPI_Finalize());

    if (commRank == 0)
    {
        cout << "PASSED\n";
    }

    return 0;
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    cout << "Test FAILED\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}
