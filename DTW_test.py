import torch
from soft_dtw import SoftDTW
from pytorch_ssim import SSIM
from predictor_env_wrapper import DreamGazeboEnv
from DDPG_IL import DDPG_IL
ssim = SSIM(window_size=10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.no_grad:
    env = DreamGazeboEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = torch.tensor(env.action_space.high, dtype=torch.float, device=device)
    agent = DDPG_IL(state_dim, action_dim, max_action)
    agent.load() # TODO
    # compute DTW score for each expert trajectory in testing set
    for sequence in env.test_loader:
        state, action, next_state, done = sequence  # (s, a, s', d) tuple of a whole trajectory
        state, action, next_state, done = state.squeeze(0), action.squeeze(0), next_state.squeeze(0), done.squeeze(0)  # removing batch dim
        expert = state
        query = [state[0]]
        env.state = state[0]
        traj_len = len(state)
        for i in range(1, traj_len):
            action = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            query.append(next_state)
            state = next_state

        # Compute DTW between expert and query
        dist_mat = [[None] * len(query)] * len(expert)
        for i in range(len(expert)):
            for j in range(len(query)):
                dist_mat[i][j] = 1 - ssim(expert[i], query[j])  # convert from similarity to distance
        dist_mat = torch.tensor(dist_mat, dtype=torch.float, device=device)
        soft_DTW = SoftDTW()
        DTW_score = soft_DTW(dist_mat)
        print(f"DTW_score: {DTW_score}")
