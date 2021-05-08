"""Soft Q Imitation Learning implemented with A2C"""
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device('cuda')

class Actor(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=256):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
        )
        # the following together outputs (batch_size, 2, num_actions), each pair (mu, sigma_sq) for each action
        # mus(means) and sigma_sqs(variances=standard deviation^2's) represents the parameters for a Gaussian distribution
        self.linear_mus = nn.Linear(hidden_size, num_actions)
        self.linear_sigma = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = self.layers(x)
        mu = self.linear_mus(x)
        log_sigma = self.linear_sigma(x)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)  # We limit the variance by forcing within a range of -20,2
        return mu, log_sigma.exp()


class Critic(nn.Module):
    def __init__(self, input_shape, hidden_size=256):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Critic represents V(s) = scalar,
        )
    
    def forward(self, state):
        x = self.layers(state)
        return x


class PolicyGradientAlgorithm(object):
    def __init__(self, env, num_actions, obs_shape, algo="REINFORCE", env_type="inverted", resume=False, test=False):
        self.env = env
        self.num_actions = num_actions
        self.input_shape = obs_shape
        self.algo = algo
        self.test = test
        self.env_type = env_type
        self.save_dir = "/content/drive/MyDrive/Colab Notebooks/525Program2/saved"
        self.weights_file = 'weights.pth'
        self.checkpoint_file = 'checkpoint.tar'
        self.timestep = 0
        self.total_timestep = 0
        self.episode = 0
        self.sum_reward = 0
        self.state = None
        self.next_state = None
        self.reward = 0
        self.done = False
        self.action = 0
        self.negative_log_likelihood = None
        self.stop_learning = False
        self.sum_rewards = []
        # DRL algorithm settings
        self.max_episodes = 4000  # DQN
        self.minibatch_size = 1000
        self.horizon = 32  # TODO 32
        self.discount = 0.99
        self.epochs = 1  # 5 does not work with various lrs. overfits
        self.target_update_freq = 10
        # DL training settings
        self.lr = 3e-4 if env_type == "inverted" else 5e-5 # 3e-4 inverted RE, 5e-5 half RE
        self.c_lr = 3e-3  # 1e-3
        self.max_grad_norm = 5  # 1
        self.actor = Actor(self.input_shape, num_actions)
        self.actor.to(device)
        self.critic = Critic(self.input_shape)
        self.critic.to(device)
        self.target_model = Actor(self.input_shape, num_actions)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.actor.state_dict())
        self.optimizer = optim.Adam(self.actor.parameters(), self.lr)
        self.c_optimizer = optim.Adam(self.critic.parameters(), self.c_lr)
        self.trainLoss = []
        self.c_trainLoss = []
    
    def get_action(self, state):
        mus, sigmas = self.actor(state)
        if self.num_actions == 1:
            dist = Normal(mus, sigmas)
        else:
            covar_mat = torch.diag(sigmas.pow(2).view(-1))
            dist = MultivariateNormal(mus, covar_mat)  # equivalent to an array of Normal's for each action
        # cannot output log_prob for each rand variable(action), only output one log prob!
        pred_action = dist.sample()  # size (num_actions,)
        self.negative_log_likelihood = -dist.log_prob(pred_action)
        self.negative_log_likelihood.requires_grad_(True)
        if self.num_actions == 1:
            pred_action = torch.tanh(pred_action).cpu().item()  # squeeze output action to (-1,1) to apply in env
            return [pred_action]
        else:
            pred_action = torch.tanh(pred_action).cpu().view(-1).numpy()  # squeeze output action to (-1,1) to apply in env
            return pred_action
    
    def restart(self):
        """ Reset env and initialize with the zeroth timestep"""
        self.state = self.env.reset()
        self.state = torch.tensor(self.state, dtype=torch.float, device=device).unsqueeze(0)  # inserts an axis representing batch of 1
    
    def random_step(self):
        self.action = self.env.action_space.sample()  # sample() returns list(float) of len 1
        self.next_state, self.reward, self.done, _ = self.env.step(self.action)
        self.next_state = torch.tensor(self.next_state, dtype=torch.float, device=device).unsqueeze(0)
    
    def e_policy_step(self):
        self.action = self.get_action(self.state)
        self.next_state, self.reward, self.done, _ = self.env.step(self.action)
        self.next_state = torch.tensor(self.next_state, dtype=torch.float, device=device).unsqueeze(0)
    
    def A2C_algorithm(self):
        def A2C_grad_desc():
            states, actions, rewards, next_states, negative_log_likelihoods = zip(*self.trajectory)
            batch_states, batch_next_states = torch.cat(states), torch.cat(next_states)
            batch_rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(-1)  # append a dimension
            
            # training critic
            # https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256/2
            # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
            # explains in-place operations and why they cause problem. Here my Actor/Critic both have in-place operations. Therefore copy states/next_states
            v_states = self.critic(batch_states)
            ys = batch_rewards + self.discount * self.critic(batch_next_states)  # 1-step TD(0) target
            # ys = returns.unsqueeze(-1)  # MC target
            c_loss = F.mse_loss(v_states, ys)  # loss function: output and target should have the same dimensions
            self.c_optimizer.zero_grad()
            c_loss.backward()  # loss.backward() expect loss to be a single scalar value
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # clipping gradient really helps!
            self.c_optimizer.step()
            self.c_trainLoss.append(c_loss.item())
            
            # training actor
            #TODO Target net? Replay buffer?
            batch_states, batch_next_states = torch.cat(states), torch.cat(next_states)
            batch_rewards = torch.tensor(rewards, dtype=torch.float, device=device).unsqueeze(-1)  # append a dimension
            batch_nlls = torch.cat(negative_log_likelihoods)  # requires_grad_(True) nor retain_grad() helps training here
            advantage = batch_rewards + self.discount * self.critic(batch_next_states) - self.critic(batch_states)
            weighted_nll = torch.mul(batch_nlls, advantage)
            loss = torch.sum(weighted_nll)  # with mean instead of sum, actor does not learn
            self.optimizer.zero_grad()
            loss.backward()  # loss.backward() expect loss to be a single scalar value
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # clipping gradient really helps!
            self.optimizer.step()
            self.trainLoss.append(loss.item())
            self.trajectory = []
        
        print("Running A2C Algorithm")
        self.restart()
        while self.episode < self.max_episodes:
            # An episode starts
            start_time = time.time()
            self.trajectory = []
            self.timestep = 0
            self.sum_reward = 0
            self.done = False
            # Generate an episode
            while not self.done:
                # A timestep starts
                self.e_policy_step()
                self.sum_reward += self.reward
                if not self.done:
                    self.trajectory.append((self.state, self.action[0], self.reward, self.next_state, self.negative_log_likelihood))
                    self.state = self.next_state
                    self.timestep += 1
                    self.total_timestep += 1
                else:  # done, an episode is about to end
                    self.trajectory.append((self.state, self.action[0], self.reward, self.next_state, self.negative_log_likelihood))
                    self.restart()
                    self.sum_rewards.append(self.sum_reward)
                    print(
                        "Episode {}, length: {} timesteps, reward: {}, moving average reward: {}, time used: {}".format(
                            self.episode, self.timestep + 1, self.sum_reward,
                            np.mean(self.sum_rewards[-10:]),
                                          time.time() - start_time))
                # end of timestep
                # apply horizon when collecting episode
                # batch of length-horizon(32)
                if self.timestep % 32 and not self.done:  # if done, the call outside will get traj of len 0
                    A2C_grad_desc()
            # another one at the end of episode
            A2C_grad_desc()
            self.episode += 1
    
    def plot(self):
        num_plots = 2 if self.algo != "A2C" else 3
        N = len(self.sum_rewards) // 20  # size of recent numbers to be averaged
        plt.figure(figsize=(20, 15))
        plt.subplot(num_plots, 1, 1)
        plt.plot(self.sum_rewards, label="reward")
        plt.plot(np.arange(N, len(self.sum_rewards) + 1),
                 np.convolve(self.sum_rewards, np.ones(N) / N, mode='valid'), color='r',
                 label="moving average of reward")
        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('Training Episodes')
        
        N = len(self.trainLoss) // 20  # size of recent numbers to be averaged
        plt.subplot(num_plots, 1, 2)
        plt.plot(self.trainLoss, label="loss for Actor")
        plt.plot(np.arange(N, len(self.trainLoss) + 1),
                 np.convolve(self.trainLoss, np.ones(N) / N, mode='valid'), color='r',
                 label="moving average of loss for Actor")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        if self.algo == "A2C":
            plt.subplot(num_plots, 1, 3)
            plt.plot(self.c_trainLoss, label="loss for Critic")
            plt.plot(np.arange(N, len(self.c_trainLoss) + 1),
                     np.convolve(self.c_trainLoss, np.ones(N) / N, mode='valid'), color='g',
                     label="moving average of loss for Critic")
            plt.xlabel("Total Timesteps")
            plt.ylabel("Loss")
            plt.legend()
        plt.show()
    
    def test_(self):
        """ Test the trained DQN in the gym environment. Run final policy 10 times, compute average return of each episode
            Call with resume=True"""
        self.restart()
        sum_rewards = []
        for _ in range(10):
            self.timestep = 0
            self.sum_reward = 0
            # self.restart()
            self.done = False
            while not self.done:
                self.e_policy_step()
                self.sum_reward += self.reward
                if not self.done:
                    self.state = self.next_state
                    self.timestep += 1
                    self.total_timestep += 1
                else:  # done, an episode is about to end
                    self.restart()
            sum_rewards.append(self.sum_reward)
            print(f"Episode finished with return: {self.sum_reward}")
        print(f"Average return of 10 episodes: {np.average(sum_rewards)}")
        self.env.close()
