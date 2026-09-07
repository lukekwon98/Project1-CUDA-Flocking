**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

## Demo

**Configuration:** 100,000 Boids · Coherent Grid · Block Size 128

### Screenshot

<img width="1276" height="720" alt="100,000 Boids - Coherent Grid - Block Size 128" src="https://github.com/user-attachments/assets/cbab3e6c-2302-44f6-9a08-6b78bf8217e3" />


### GIF

<img width="1280" height="720" alt="100,000 Boids - Coherent Grid - Block Size 128" src="https://github.com/user-attachments/assets/c85d4a68-24b1-4de6-a508-8df1644386da" />


## Performance Analysis
All performance tests were run in Release mode with V-Sync disabled. For each configuration, the simulation was allowed to run for 200 warm-up frames before collecting measurements, reducing startup and initialization effects. After the warm-up, performance was measured over 1,000 simulation frames and the average FPS was calculated from the total elapsed time.

### 1. Effect of Boid Count

#### Per Boid Count - Visualization On/Off

| Boid Count - Visualization On | Boid Count - Visualization Off |
|---|---|
| <img width="600" alt="Performance Per Boid Count - Visualization On" src="https://github.com/user-attachments/assets/79a38071-a725-48e9-bb4c-ce842e3686b2" /> | <img width="600" alt="Performance Per Boid Count - Visualization Off" src="https://github.com/user-attachments/assets/c5806fe1-bd3d-4ed0-920a-cd6027b384c5" /> |

Increasing the number of boids generally decreased performance for all three implementations. 
- The naive implementation showed the largest performance decrease because every boid checks every other boid when computing its, which caused the amount of neighbor search work to grow quadratically with the number of boids.
- The scattered uniform grid scaled noticeably better because each boid only examines particles in nearby grid cells rather than the entire simulation. However, the particle data being accessed is scattered in memory, so the neighbor search still sufferes from less efficient memory access
- The coherent uniform grid also scaled much better than the naive implementation and performed best at larger boid counts. It performs the same spatial culling as the scattered grid but rearranges particles so that boids in the same grid cell are continguous in memory, which improves memory localitly and cache coalescing behavior.
- An interesting observation was that the performance for 100,000 boids was better than 50,000 boids for the coherent grid implementation, the cause of which is under yet to be determined.

### 2. Effect of Block Count and Block Size

#### Per Block Size - Visualization Off

| Block Size - Visualization Off |
|:---:|
| <img width="500" alt="Performance Per Block Size - Visualization Off" src="https://github.com/user-attachments/assets/cbf9c46c-e526-4410-9232-630774515f9d" /> |

Across all three implementations, performance improved as block size increased from 32 to about 128–256 threads per block, then leveled off or declined for larger blocks. This lines up well with the occupancy measurements on the RTX 3060. At 32 threads per block, all three main kernels reached only about 33% occupancy, and at 64 threads they reached about 67%. The naive kernel reached 100% occupancy at block sizes 128, 256, and 512, while the scattered and coherent neighbor-search kernels peaked at about 83% occupancy at 128 and 256 threads per block. At 512 and 1024 threads, the grid kernels dropped back to about 67% occupancy.

| Naive Brute Force Kernel Occupancy | Scattered Uniform Grid Kernel Occupancy | Coherent Uniform Grid Kernel Occupancy |
|---|---|---|
| <img width="731" height="230" alt="Screenshot 2026-09-06 234243" src="https://github.com/user-attachments/assets/1d163e97-2119-41c6-a941-82c9d5d3c954" /> | <img width="733" height="229" alt="Screenshot 2026-09-06 234253" src="https://github.com/user-attachments/assets/57f845f9-5c14-4e59-af7c-1a8d8f28ae73" /> | <img width="727" height="227" alt="Screenshot 2026-09-06 234304" src="https://github.com/user-attachments/assets/c975be6d-8ba2-4456-b2c0-e2d9284acee1" /> |


The measured FPS follows this pattern: 32-thread blocks performed worst, 128–256 threads gave the best results, and performance declined again at 512–1024 for the grid implementations. This suggests that the mid-sized blocks provided the best balance of active warps and resident blocks per SM, allowing the GPU to hide latency more effectively. The profiling also showed that the scattered and coherent kernels used slightly more registers per thread, 42 and 41 registers, respectively, compared with 35 registers for the naive kernel, which helps explain why the grid kernels could not reach the same 100% occupancy as the naive kernel at 128–512 threads. For a fixed number of boids, the block count is determined by the block size, approximately N / blockSize. Increasing the block size therefore decreases the number of blocks. Too few blocks can reduce the scheduler's ability to keep the GPU occupied, while very small blocks may not provide enough active threads per block to efficiently utilize the SMs.

### 3. Coherent vs Scattered Uniform Grid
The coherent uniform grid generally provided an improvement over the scattered uniform grid, which was the expected result.

In the scattered implementation, particles belonging to the same cell are represented by contiguous indices, but those indices still point to position and velocity data located at unrelated locations in memory. The coherent implementation additionally rearranges the actual position and velocity arrays so that particles in the same cell are contiguous. This improves spatial locality and makes neighboring GPU threads more likely to access nearby memory locations. The improvement may be relatively small, or the scattered uniform grid may perform even better at low boid counts because the coherent implementation also has the additional cost of rearranging the particle arrays every simulation step. As the workload grows, the improved memory behavior becomes more valuable.


### 8 Cells vs 27 Cells

| 8 Cells | 27 Cells |
|---|---|
| <img width="403" height="157" alt="Screenshot 2026-09-06 235013" src="https://github.com/user-attachments/assets/aa608d25-bcf3-42ef-b8f3-98c0c3418af2" /> | <img width="397" height="156" alt="Screenshot 2026-09-06 235056" src="https://github.com/user-attachments/assets/c23a7661-f38b-46b6-8aa8-0a384bd6c7fe" /> |

*Performance comparison using the Coherent Grid implementation with 500,000 boids and a block size of 128.*

The 27 cell search performed better, reaching roughly 420 FPS, compared with about 360 FPS for the 8 cell search. Even though checking 27 neighboring cells requires more grid-cell lookups, the smaller cell width gives the grid finer spatial granularity. As a result, each cell contains fewer boids on average, so each boid performs fewer unnecessary distance comparisons against particles that are actually too far away to influence it. In this case, the savings from reducing those extra particle comparisons outweighed the additional overhead of visiting more grid cells.

## Implemented Extra Credit
Grid-Looping Optimization
