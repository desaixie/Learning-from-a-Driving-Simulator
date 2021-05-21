# Learning-from-a-Driving-Simulator
## Zero-shot Model-based Imitation Learning with Partial Trajectory and Sparse Rewards

### Abstract
In this project, I attempt to apply Imitation Learning (IL) to train an agent to tackle the planning problem in Autonomous Driving (AD).  
With this prolem being narrowed down to the Zero-shot Model-based IL setting, I achieve IL with the sparse-rewards reward function, which has shown great performance with raw-pixel state space, and invented the technique Partial Trajectory to accomodate the limitations of the state-transition model in hallucinating long trajectories in zero-shot model-based learning. As a result, I achieved comparable performance with the Behavioral Cloning agent.  

### References
`future-image-similarity/` is adapted from the Github repo for the paper [*Model-based Behavioral Cloning with Future Image Similarity Learning*](https://github.com/anwu21/future-image-similarity)  
`pytorch_ssim/` is adapted from the Github [repo](https://github.com/Po-Hsun-Su/pytorch-ssim)  
`soft_dtw.py` is adapted from the Github [repo](https://github.com/Sleepwalking/pytorch-softdtw)  
The rest of the .py files is my own work.  
