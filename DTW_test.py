"""
Run five DTW tests with the trained agents repeatively select on-policy actions in DreamGazeboEnv
First three are three DDPG_SQIL models trained differently
Last two are MBC, including expert action in candidates or not
"""
import torch
from soft_dtw import SoftDTW
from pytorch_ssim import SSIM
from predictor_env_wrapper import DreamGazeboEnv
from gazebo_env_wrapper import SimGazeboEnv
from gaz_IL import Gazebo
from DDPG_SQIL import DDPG_SQIL
from MBC import MBC_Agent

ssim = SSIM(window_size=10)  # TODO
device = 'cuda' if torch.cuda.is_available() else 'cpu'
testMode = "simulation"  # simulation or dream
if testMode == "dream":
    with torch.no_grad():
        env = DreamGazeboEnv()
        saved_paths = ["logs/expDDPG_IL.pyDreamGazebo/" + t for t in ["default/",  # default
                                                                      "clipgrad/",  # with clip_grad_norm
                                                                      "clipgrad_partialtraj/"]]  # with clip_grad_norm, max_timestep=7
        avg_DTW_scores = []
        for i_test in range(5):  # run 5 DTW tests
            if i_test < 3:
                state_dim = env.observation_space.shape
                action_dim = env.action_space.shape[0]
                max_action = torch.tensor(env.action_space.high, dtype=torch.float, device=device)
                agent = DDPG_SQIL(state_dim, action_dim, max_action)
                agent.load(saved_paths[i_test])
                agent.actor.eval()
                agent.critic.eval()
            else:
                agent = MBC_Agent(env)
                agent.critic.eval()
                remove_expert_action = False if i_test == 3 else True
            # compute DTW score for each expert trajectory in testing set
            avg_DTW_score = 0
            for sequence in env.test_loader:
                states, actions, _, _ = sequence  # (s, a, s', d) tuple of a whole trajectory
                states, actions = states.squeeze(0).float().to(device), actions.squeeze(0).float().to(device)
                expert = states
                state = states[0].unsqueeze(0)
                query = [state]
                env.state = state
                traj_len = len(expert)
                for i in range(1, traj_len):
                    if i_test < 3:
                        action = agent.select_action(state)
                    else:
                        candidates = actions[i, :, :]  # traj_len, N_candidates, action_dim
                        if remove_expert_action:
                            candidates = candidates[1:]  # 0th expert action is not provided
                        action = agent.select_action(state, candidates)
                    next_state, _, done, _ = env.step(action)
                    query.append(next_state)
                    state = next_state

                # Compute DTW between expert and query
                dist_mat = [[None] * len(query)] * len(expert)
                for i in range(len(expert)):
                    for j in range(len(query)):
                        dist_mat[i][j] = 1 - ssim(expert[i].unsqueeze(0), query[j])  # convert from similarity to distance
                dist_mat = torch.tensor(dist_mat, dtype=torch.float, device=device)
                soft_DTW = SoftDTW(gamma=0.01)  # gamma -> 0, Soft -> hard/original DTW
                DTW_score = soft_DTW(dist_mat.unsqueeze(2)).mean()  # append a feature dimension, as required by soft_DTW
                avg_DTW_score += DTW_score
                # print(f"DTW_score: {DTW_score}")
            avg_DTW_score /= len(env.test_loader)
            print(f"Test {i_test} average DTW score: {avg_DTW_score}")
            avg_DTW_scores.append(avg_DTW_score)
    # test 0 36.2705039  default
    # test 1 36.4431496  clip_grad
    # test 2 34.6599686  clip_grad + max_timestep=7
    # test 3 33.9678452  MBC
    # test 4 34.3164032  MBC without expert action in cadidates

else:  # simulation DTW test
    nCandidates = 16  # num candidates for MBC_Agent
    with torch.no_grad():
        env = SimGazeboEnv()
        saved_paths = ["logs/expDDPG_IL.pyDreamGazebo/" + t for t in ["2021-05-18 11:10:02/",  # default
                                                                      "2021-05-18 12:46:48/",  # with clip_grad_norm
                                                                      "2021-05-19 13:29:45/"]]  # with clip_grad_norm, max_timestep=7
        avg_DTW_scores = []
        for i_test in range(4):  # run DTW test for each agent
            if i_test < 3:
                state_dim = env.observation_space.shape
                action_dim = env.action_space.shape[0]
                max_action = torch.tensor(env.action_space.high, dtype=torch.float, device=device)
                agent = DDPG_SQIL(state_dim, action_dim, max_action)
                agent.load(saved_paths[i_test])
                agent.actor.eval()
                agent.critic.eval()
            else:
                agent = MBC_Agent(env)
                agent.critic.eval()
            # Each agent run 10 trajectories in the simulation. For each trajectory, find its min DTW
            # score against all expert trajectories in testing set. Compute average of 10 min DTW scores.
            for i in range(10):
                state, _, done, _ = env.reset()
                while not done:
                    if i_test < 3:
                        action = agent.select_action(state)
                    else:
                        candidates = []
                        for k in range(nCandidates):
                            curr_angle = None  # TODO get from simulation
                            R = None  # TODO randomly generate?
                            dx, dy, d_phi = Gazebo.svg_gen_diff_pose_value(curr_angle, R)
                            candidates.append(torch.tensor([dx, dy, d_phi], dtype=torch.float))  # don't make tensors on GPU
                        action = agent.select_action(state, candidates)
                    state, _, done, _ = env.step(action)
                # TODO Done, load trajectory Odom file, compute DTW score
