**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

## Demo

**Configuration:** 100,000 Boids · Coherent Grid · Block Size 128

### Screenshot

<img width="1276" height="720" alt="100,000 Boids - Coherent Grid - Block Size 128" src="https://github.com/user-attachments/assets/cbab3e6c-2302-44f6-9a08-6b78bf8217e3" />

*100,000 boids using the coherent uniform grid implementation with a block size of 128.*

### GIF

<img width="1280" height="720" alt="100,000 Boids - Coherent Grid - Block Size 128" src="https://github.com/user-attachments/assets/c85d4a68-24b1-4de6-a508-8df1644386da" />

*Real-time flocking simulation with 100,000 boids using the coherent uniform grid implementation and a block size of 128.*
### Performance Analysis
#### Per Boid Count - Visualization On/Off

| Boid Count - Visualization On | Boid Count - Visualization Off |
|---|---|
| <img width="600" alt="Performance Per Boid Count - Visualization On" src="https://github.com/user-attachments/assets/79a38071-a725-48e9-bb4c-ce842e3686b2" /> | <img width="600" alt="Performance Per Boid Count - Visualization Off" src="https://github.com/user-attachments/assets/c5806fe1-bd3d-4ed0-920a-cd6027b384c5" /> |

#### Per Block Size

