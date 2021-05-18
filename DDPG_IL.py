"""
DDPG with SQIL reward function, intended to use with future-image-similarity image predictor as the
state-transition model and its Gazebo simulation dataset as expert trajectories
"""
import argparse
from itertools import count

import os, sys, random, time
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

#import from code in this project
from predictor_env_wrapper import DreamGazeboEnv
from future_image_similarity import utils

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient, 0.001 in DDPG paper
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor, 0.99 in DDPG paper
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size, 1e6 in DDPG paper
parser.add_argument('--batch_size', default=100, type=int) # mini batch size, 64 for low-dim and 16 for pixel in DDPG paper
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--max_grad_norm', default=5, type=int)
args = parser.parse_args()
print(args)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = DreamGazeboEnv()

if args.seed:
    # env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = torch.tensor(env.action_space.high, dtype=torch.float, device=device)

now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
directory = './logs/exp' + script_name + "DreamGazebo/" + now + "/"

class ReplayBuffer():
    """
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, action, reward, next_state, done)
    """
    def __init__(self, max_size=args.capacity):
        self.storage = deque(maxlen=max_size)
        self.max_size = max_size
        self.ptr = 0
    
    def push(self, data):
        """ data: tuple of (state, action, reward, next_state, done)
            state, next_state are batch_size=1 tensors, the rest are scalar, floats"""
        self.storage.append([d if torch.is_tensor(d) else torch.tensor(d, dtype=torch.float) for d in data])
    
    def sample(self, batch_size):
        minibatch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        # stacking actions, rewards, dones which have no 0th batch dim; cating states, which have
        states, actions, rewards, next_states, dones = torch.cat(states).to(device), torch.stack(actions).to(device), torch.stack(rewards).unsqueeze(1).to(device), torch.cat(next_states).to(device), torch.stack(dones).unsqueeze(1).to(device)  # cannot cat 0-D tensors, stack them to 1D. stacking states/actions to insert batch_size dim before. unsqueeze to append dim
        return states, actions, rewards, next_states, dones

    def save(self, path):
        torch.save(self.storage, path)

    def load(self, path):
        self.storage = torch.load(path)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.W, self.H, self.C = state_dim
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(self.C),
    
            nn.Conv2d(in_channels=self.C, out_channels=32, kernel_size=8, stride=4),  # in/output shape: (batch_size, in/out channels, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(32, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # outputs (1, 64, 7, 7)
        )
        # automatically determine dimension out flattened CNN output
        rand = torch.randn((1, self.C, self.H, self.W))  # (N, C, H, W)
        self.linear_dim = torch.flatten(self.feature_extractor(rand), start_dim=1).size()[1]
        self.actor = nn.Sequential(
            nn.Linear(self.linear_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, action_dim),
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.actor(x)  # (batch_size, num_actions)
        actions = self.max_action * torch.tanh(x)  # first constraint to (-1,1), then scale to max_action
        return actions


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.W, self.H, self.C = state_dim
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(self.C),
    
            nn.Conv2d(in_channels=self.C, out_channels=32, kernel_size=8, stride=4),  # in/output shape: (batch_size, in/out channels, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(32, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # outputs (1, 64, 7, 7)
        )
        # automatically determine dimension out flattened CNN output
        rand = torch.randn((1, self.C, self.H, self.W))  # (N, C, H, W)
        self.linear_dim = torch.flatten(self.feature_extractor(rand), start_dim=1).size()[1]
        self.actor = nn.Sequential(
            nn.Linear(self.linear_dim, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, action_dim)
        )
        self.l1 = nn.Linear(self.linear_dim, 200)
        self.b1 = nn.BatchNorm1d(200)
        self.l2 = nn.Linear(200, 200)
        self.l2action = nn.Linear(action_dim, 200)
        self.l3 = nn.Linear(200, 1)  # output
    
    def forward(self, x, action):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.b1(self.l1(x)))
        x = F.relu(self.l2(x) + self.l2action(action))
        x = self.l3(x)
        return x


class DDPG_IL(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)
        
        self.replay_buffer = ReplayBuffer()
        self.expert_buffer = ReplayBuffer(100000)
        self.expert_buffer.load("logs/expert_buffer.pth")
        self.writer = SummaryWriter(directory)
        # $ tensorboard --logdir='logs/expDDPG_IL.pyDreamGazebo.' --port=16007 &  # on server
        # $ ssh -NfL 16007:localhost:16007 desaixie@40.117.41.247 -i ~/.ssh/Azure-gpu_key_0416.pem  # local machine
        # visit http://localhost:16007/ on local machine
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.sum_rewards, self.a_trainLoss, self.c_trainLoss = [], [], []
        
        self.batch_size = args.batch_size
        self.train_loader = env.train_loader
        self.test_loader = env.test_loader

        # # filling expert_buffer
        # for sequence in self.train_loader:
        #     state, action, next_state, done = sequence  # (s, a, s', d) tuple of a whole trajectory
        #     state, action, next_state, done = state.squeeze(0), action.squeeze(0), next_state.squeeze(0), done.squeeze(0)  # removing batch dim
        #     reward = 1  # SQIL reward of expert trajectory
        #     traj_len = len(state)
        #     for i in range(traj_len):
        #         # self.expert_buffer.push((torch.tensor(state[i], dtype=torch.float), torch.tensor(action[i], dtype=torch.float), reward, torch.tensor(next_state[i], dtype=torch.float), done[i]))
        #         self.expert_buffer.push((state[i].clone().unsqueeze(0), action[i].clone(), reward, next_state[i].clone().unsqueeze(0), done[i].clone()))
        # self.expert_buffer.save("logs/expert_buffer.pth")
        # print(f"expert_buffer len {len(self.expert_buffer.storage)}")  # total of 8162 (s, a, r, s', d) tuples
        
    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        self.actor.eval()
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self):
        self.actor.train()
        self.critic.train()
        for it in range(args.update_iteration):
            # Sample replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample(args.batch_size)
            e_state, e_action, e_reward, e_next_state, e_done = self.expert_buffer.sample(args.batch_size)
            done = 1-done  # 0 if done (so multiply done = 0 below), 1 else
            e_done = 1-e_done
            full_state, full_action, full_reward, full_next_state, full_done = torch.cat([state, e_state]), torch.cat([action, e_action]), torch.cat([reward, e_reward]), torch.cat([next_state, e_next_state]), torch.cat([done, e_done])
            
            # train critic with half policy samples half expert samples. Q(s,a) critic is only used to train the actor
            target_Q = self.critic_target(full_next_state, self.actor_target(full_next_state))
            target_Q = full_reward + (full_done * args.gamma * target_Q).detach()
            current_Q = self.critic(full_state, full_action)
            
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            self.critic_optimizer.step()
            self.c_trainLoss.append(critic_loss.item())
            
            # train actor
            actor_loss = -self.critic(full_state, self.actor(full_state)).mean()  # critic acts as the loss function.
            # TODO WHY NEGATIVE??
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.actor_optimizer.step()
            self.a_trainLoss.append(actor_loss.item())
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
    
    
    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")
    
    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
        
    def plot(self):
        num_plots = 3
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
    
        N = len(self.a_trainLoss) // 20  # size of recent numbers to be averaged
        plt.subplot(num_plots, 1, 2)
        plt.plot(self.a_trainLoss, label="loss for Actor")
        plt.plot(np.arange(N, len(self.a_trainLoss) + 1),
                 np.convolve(self.a_trainLoss, np.ones(N) / N, mode='valid'), color='r',
                 label="moving average of loss for Actor")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(num_plots, 1, 3)
        plt.plot(self.c_trainLoss, label="loss for Critic")
        plt.plot(np.arange(N, len(self.c_trainLoss) + 1),
                 np.convolve(self.c_trainLoss, np.ones(N) / N, mode='valid'), color='g',
                 label="moving average of loss for Critic")
        plt.xlabel("Total Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def plot_trajectory(self, trajectory, episode):
        """ trajectory is a list of states of DreamGazeboEnv, which is essentially imagined robot observation at each timestep"""
        to_plot = []
        ncol = 5
        i = 0
        while i < len(trajectory):
            row = []
            for _ in range(ncol):
                row.append(trajectory[i])
                i += 1
                if i == len(trajectory):
                    break
            if len(row) == ncol:  # discard row not filled
                to_plot.append(row)
        fname = f"logs/trajectory/DDPG_IL/{episode}.png"
        utils.save_tensors_image(fname, to_plot)

# def main():
agent = DDPG_IL(state_dim, action_dim, max_action)
ep_r = 0
if args.mode == 'test':
    agent.load()
    for i in range(args.test_iteration):
        state = env.reset()
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(np.float32(action))
            ep_r += reward
            env.render()
            if done or t >= args.max_length_of_trajectory:
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                ep_r = 0
                break
            state = next_state

elif args.mode == 'train':
    if args.load: agent.load()
    total_step = 0
    # initialize ReplayBuffer
    state = env.reset()  # (1, 64, 64, 3)
    for i in range(args.batch_size):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step(action)
        reward = 0  # SQIL reward
        agent.replay_buffer.push((state, action, reward, next_state, np.float(done)))
        state = next_state
        if done:
            state = env.reset()
            
    # run episodes
    for i in range(args.max_episode):
        start_time = time.time()
        total_reward = 0
        step = 0
        state = env.reset()
        trajectory = [state.squeeze(dim=0)]
        actions = []  # check actions taken in episode to debug

        # a timestep
        for t in count():
            action = agent.select_action(state)
            # action = action + exploration_noise
            action = (action + np.random.normal(0, args.exploration_noise * env.action_space.high, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)
            
            next_state, _, done, _ = env.step(action)
            if args.render and i >= args.render_interval : env.render()
            reward = 0  # SQIL reward
            agent.replay_buffer.push((state, action, reward, next_state, np.float(done)))
            state = next_state
            trajectory.append(state.squeeze(dim=0))  # remove batch dimension to be plotted
            actions.append(action)
            if done:
                break
            step += 1
            total_reward += reward
            
        total_step += step+1
        agent.sum_rewards.append(total_reward)
        agent.update()  # train agent at the end of the episode
        print("Episode {}, length: {} timesteps, reward: {:.1f}, moving average reward: {:.1f}, time used: {:.1f}".format(
                i, step, total_reward, np.mean(agent.sum_rewards[-10:]), time.time() - start_time))
        
        if i % args.log_interval == 0:
            agent.save()
            agent.plot_trajectory(trajectory, i)
    agent.plot()
else:
    raise NameError("mode wrong!!!")

# if __name__ == '__main__':
#     main()