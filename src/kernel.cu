#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.1 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f
#define maxDistanceScale 2.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;


int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_sortedPos;
glm::vec3* dev_sortedVel1;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = maxDistanceScale * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_sortedPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sortedPos failed!");

  cudaMalloc((void**)&dev_sortedVel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_sortedVel1 failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernSortPosVel1(int N, int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel1, glm::vec3* sortedPos, glm::vec3* sortedVel1) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index >= N) return;

    sortedPos[index] = pos[particleArrayIndices[index]];
    sortedVel1[index] = vel1[particleArrayIndices[index]];
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.1 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 perceived_center(0.f);
    glm::vec3 posSelf = pos[iSelf];
    glm::vec3 c(0.f);
    glm::vec3 perceived_velocity(0.f);

    glm::vec3 v1(0.f);
    glm::vec3 v2(0.f);
    glm::vec3 v3(0.f);
     
    int num_neighbors = 0;
    int num_neighbors3 = 0;
    for (int i = 0; i < N; i++) {
        glm::vec3 p = pos[i];

        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (i != iSelf && glm::distance(p, posSelf) < rule1Distance) {
            perceived_center += p;
            num_neighbors++;
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (i != iSelf && glm::distance(p, posSelf) < rule2Distance) {
            c -= (p - posSelf);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (i != iSelf && glm::distance(p, posSelf) < rule3Distance) {
            perceived_velocity += vel[i];
            num_neighbors3++;
        }

    }

    // Rule 1
    if (num_neighbors != 0) {
        perceived_center /= num_neighbors;
        v1 = (perceived_center - posSelf) * rule1Scale;
    }

    // Rule 2
    v2 = c * rule2Scale;

    // Rule 3
    if (num_neighbors3 != 0) {
        perceived_velocity /= num_neighbors3;
        v3 = perceived_velocity * rule3Scale;
    }

  return v1 + v2 + v3;
}

/**
* TODO-1.1 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1? -> because pingpong?
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= N) {
        return;
    }

    glm::vec3 result = vel1[i] + computeVelocityChange(N, i, pos, vel1);
    float velMag = glm::length(result);
    if (velMag > maxSpeed) result = result/velMag * maxSpeed;
    vel2[i] = result;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
     
    // - Label each boid with the index of its grid cell.
    glm::vec3 p = pos[index];
    int iX = std::floor((p.x - gridMin.x) * inverseCellWidth);
    int iY = std::floor((p.y - gridMin.y) * inverseCellWidth);
    int iZ = std::floor((p.z - gridMin.z) * inverseCellWidth);
  
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index; //same index will be used to access pos, vel1, vel2
    gridIndices[index] = iX + iY * gridResolution + iZ * gridResolution * gridResolution;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    
    if (index == 0) {
        gridCellStartIndices[particleGridIndices[index]] = index;
    }
    else if (particleGridIndices[index - 1] != particleGridIndices[index]) {
        gridCellStartIndices[particleGridIndices[index]] = index;
    }

    if (index == N - 1) {
        gridCellEndIndices[particleGridIndices[index]] = index;
    }
    else if (particleGridIndices[index + 1] != particleGridIndices[index]) {
        gridCellEndIndices[particleGridIndices[index]] = index;
    }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    int num_neighbors = 0;
    int num_neighbors3 = 0;
    glm::vec3 perceived_center(0.f);
    glm::vec3 c(0.f);
    glm::vec3 perceived_velocity(0.f);

    glm::vec3 v1(0.f);
    glm::vec3 v2(0.f);
    glm::vec3 v3(0.f);

  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
    glm::vec3 p = pos[index];

    float maxDist = rule1Distance;
    if (rule2Distance > maxDist) maxDist = rule2Distance;
    if (rule3Distance > maxDist) maxDist = rule3Distance;

    glm::vec3 minCell = p - glm::vec3(maxDist);
    glm::vec3 maxCell = p + glm::vec3(maxDist);

    minCell = glm::floor((minCell - gridMin) * inverseCellWidth);
    maxCell = glm::floor((maxCell - gridMin) * inverseCellWidth);

    for (int z = minCell.z; z <= maxCell.z; z++) {
        if (z < 0 || z >= gridResolution) continue;

        for (int y = minCell.y; y <= maxCell.y; y++) {
            if (y < 0 || y >= gridResolution) continue;
            
            for (int x = minCell.x; x <= maxCell.x; x++) {
                if (x < 0 || x >= gridResolution) continue;
                
                int neighborGridId = x + y * gridResolution + z * gridResolution* gridResolution;
                int startIdx = gridCellStartIndices[neighborGridId];
                int endIdx = gridCellEndIndices[neighborGridId];

                if (startIdx == -1 || endIdx == -1) continue;

                for (int sortedGridCellIdx = startIdx; sortedGridCellIdx <= endIdx; sortedGridCellIdx++) {
                    if (particleArrayIndices[sortedGridCellIdx] != index) {
                        int boidID = particleArrayIndices[sortedGridCellIdx];
                        float dist = glm::distance(pos[boidID], p);

                        if (dist < rule1Distance) {
                            perceived_center += pos[boidID];
                            num_neighbors++;
                        }

                        if (dist < rule2Distance) {
                            c -= (pos[boidID] - p);
                        }

                        if (dist < rule3Distance) {
                            perceived_velocity += vel1[boidID];
                            num_neighbors3++;
                        }
                    }
                }
            }
        }
    }
    // Rule 1
    if (num_neighbors != 0) {
        perceived_center /= num_neighbors;
        v1 = (perceived_center - p) * rule1Scale;
    }

    // Rule 2
    v2 = c * rule2Scale;

    // Rule 3
    if (num_neighbors3 != 0) {
        perceived_velocity /= num_neighbors3;
        v3 = perceived_velocity * rule3Scale;
    }

    glm::vec3 newVel = vel1[index] + v1 + v2 + v3;
    float velMag = glm::length(newVel);
    if (velMag > maxSpeed) {
        newVel = newVel/velMag * maxSpeed;
    }

    vel2[index] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    int num_neighbors = 0;
    int num_neighbors3 = 0;
    glm::vec3 perceived_center(0.f);
    glm::vec3 c(0.f);
    glm::vec3 perceived_velocity(0.f);

    glm::vec3 v1(0.f);
    glm::vec3 v2(0.f);
    glm::vec3 v3(0.f);

  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
    glm::vec3 p = pos[index];

    float maxDist = rule1Distance;
    if (rule2Distance > maxDist) maxDist = rule2Distance;
    if (rule3Distance > maxDist) maxDist = rule3Distance;

    glm::vec3 minCell = p - glm::vec3(maxDist);
    glm::vec3 maxCell = p + glm::vec3(maxDist);

    minCell = glm::floor((minCell - gridMin) * inverseCellWidth);
    maxCell = glm::floor((maxCell - gridMin) * inverseCellWidth);

  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
    for (int z = minCell.z; z <= maxCell.z; z++) {
        if (z < 0 || z >= gridResolution) continue;

        for (int y = minCell.y; y <= maxCell.y; y++) {
            if (y < 0 || y >= gridResolution) continue;

            for (int x = minCell.x; x <= maxCell.x; x++) {
                if (x < 0 || x >= gridResolution) continue;

                int neighborGridId = x + y * gridResolution + z * gridResolution * gridResolution;
                int startIdx = gridCellStartIndices[neighborGridId];
                int endIdx = gridCellEndIndices[neighborGridId];

                if (startIdx == -1 || endIdx == -1) continue;

                for (int sortedGridCellIdx = startIdx; sortedGridCellIdx <= endIdx; sortedGridCellIdx++) {
                    if (sortedGridCellIdx != index) {
                        float dist = glm::distance(pos[sortedGridCellIdx], p);

                        if (dist < rule1Distance) {
                            perceived_center += pos[sortedGridCellIdx];
                            num_neighbors++;
                        }

                        if (dist < rule2Distance) {
                            c -= (pos[sortedGridCellIdx] - p);
                        }

                        if (dist < rule3Distance) {
                            perceived_velocity += vel1[sortedGridCellIdx];
                            num_neighbors3++;
                        }
                    }
                }
            }
        }
    }

  // - Clamp the speed change before putting the new speed in vel2
    // Rule 1
    if (num_neighbors != 0) {
        perceived_center /= num_neighbors;
        v1 = (perceived_center - p) * rule1Scale;
    }

    // Rule 2
    v2 = c * rule2Scale;

    // Rule 3
    if (num_neighbors3 != 0) {
        perceived_velocity /= num_neighbors3;
        v3 = perceived_velocity * rule3Scale;
    }

    glm::vec3 newVel = vel1[index] + v1 + v2 + v3;
    float velMag = glm::length(newVel);
    if (velMag > maxSpeed) {
        newVel = newVel / velMag * maxSpeed;
    }

    vel2[index] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    int totalBlocks = (numObjects + blockSize - 1) / blockSize;
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    kernUpdateVelocityBruteForce << < totalBlocks, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos << <totalBlocks, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    // TODO-1.2 ping-pong the velocity buffers
    std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize); 
  dim3 fullBlocksPerGridCellSort((gridCellCount + blockSize - 1) / blockSize);

  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, 
  gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  
    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    kernResetIntBuffer << <fullBlocksPerGridCellSort, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCellSort, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
    std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCellSort((gridCellCount + blockSize - 1) / blockSize);

  // - Label each particle with its array index as well as its grid index.
      //   Use 2x width grids
    kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    kernResetIntBuffer << <fullBlocksPerGridCellSort, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCellSort, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    kernSortPosVel1 << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_sortedPos, dev_sortedVel1);
  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > 
        (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_sortedPos, dev_sortedVel1, dev_vel2);

  // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_sortedPos, dev_vel2);

  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    std::swap(dev_pos, dev_sortedPos);
    std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_sortedPos);
  cudaFree(dev_sortedVel1);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}