# Learning-from-a-Driving-Simulator
## Zero-trial Model-based Imitation Learning with Partial Trajectory

### Abstract
This project is an initial attempt to apply Imitation Learning (IL) to train an agent to tackle the planning problem in Autonomous Driving (AD) or, more generally, robot control problem with visual input. 
Specifically, on a task simplified from AD, I propose a zero-trial model-based IL algorithm to train a agent to control a ground mobile robot on a target-approaching task. This algorithm includes an action-based future image predictor, an actor-critic framework, an IL reward function, and partial trajectory, an technique that I invent to accommodate the limitations of the state-transition model in hallucinating long trajectories. I discuss my reason behind using this problem setting and the advantages of my algorithm's components under this setting. Finally, I show that, experimentally, my method can achieve comparable performance with the Behavioral Cloning (BC) agent on Dynamic Time Warping tests.

### References
- `future-image-similarity/` is adapted from the Github repo for the paper [*Model-based Behavioral Cloning with Future Image Similarity Learning*](https://github.com/anwu21/future-image-similarity) (*MBC*)  
- `pytorch_ssim/` is adapted from the Github [repo](https://github.com/Po-Hsun-Su/pytorch-ssim)  
- `soft_dtw.py` is adapted from the Github [repo](https://github.com/Sleepwalking/pytorch-softdtw)  

The rest of the `.py` files is my own work.  

- `DDPG.py` is my implementation of [*DDPG*](https://arxiv.org/abs/1509.02971). It is tested in [`Gym`](https://gym.openai.com/docs/)'s InvetedPendulum-v0, HalfCheetah-v0, and CarRacing-v0.   
- `DDPG_SQIL.py` is my core algorithm. It is implemented with the *MBC* action-based future image predictor, the DDPG actor-critic framework, the [*SQIL*](https://arxiv.org/abs/1905.11108) sparse-rewards reward function, and the Partial Trajectory technique.  
- `DTW_test.py` is the code for runing [*Dynamic Time Warping*](https://arxiv.org/abs/1703.01541) (DTW) tests, comparing the trajectories hallucinated in the image predictor by running *DDPG_SQIL* and *MBC* against the expert trajectories in testing set.  
- `GazeboAPI.py` is the unfinished code. I plan to use the [`rospy`](http://wiki.ros.org/rospy) package to implement a bridge between the Gazebo simulation and my algorithm in `DDPG_SQIL.py`. Hyperparameter tuning of my algorithm will be done after implementing `GazeboAPI.py`.  
- `MBC.py` wraps *MBC*'s critic to have the same API as `DDPG_SQIL.py`.  
- `Wrapper.py` is used to run `DDPG.py` in CarRacing-v0.  
- `gaz_IL.py` rewrites the `class Gazebo` custom dataset class from `future-image-similarity/data/gaz_value.py` to use the dataset collected in Gazebo in a more general way.  
- `migrate_models.py` loads saved models from `future-image-similarity/logs` and save them with whatever pytorch version you are using.  
- `predictor_env_wrapper.py` contains `class DreamGazeboEnv`, which wraps *MBC*'s image predictor and the dataset loaded with `gaz_IL.py` as a `Gym` environment.  
