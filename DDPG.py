import argparse
from itertools import count

import os, sys, random, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) # error only
import pybulletgym  # register PyBullet enviroments with open ai gym
import pybullet
import pybullet_data

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="InvertedPendulumMuJoCoEnv-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7, dtype=torch.float, device=device) # min value

directory = './logs/exp' + script_name + args.env_name +'./'

class ReplayBuffer():
    """
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, action, reward, next_state, done)
    """
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    
    def push(self, data):
        """ data: tuple of (state, action, reward, next_state, done)"""
        tensor_data = []
        for d in data:
            tensor_data.append(torch.tensor(d, dtype=torch.float, device=device))
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = tensor_data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(tensor_data)
    
    def sample(self, batch_size):
        minibatch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, actions, rewards, next_states, dones = torch.stack(states), torch.stack(actions), torch.stack(rewards).unsqueeze(1), torch.stack(next_states), torch.stack(dones).unsqueeze(1)  # cannot cat 0-D tensors, stack them to 1D. stacking states/actions to insert batch_size dim before. unsqueeze to append dim
        return states, actions, rewards, next_states, dones


class Actor(nn.Module):
    # TODO BatchNorm
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))  # first contraint to (-1,1), then scale to max_action
        # TODO support multiple actions
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
    
    def forward(self, x, action):
        x = F.relu(self.l1(torch.cat([x, action], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.sum_rewards, self.a_trainLoss, self.c_trainLoss = [], [], []
        
    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)  # inserts an axis representing batch of 1
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self):
        for it in range(args.update_iteration):
            # Sample replay buffer
            state, action, reward, next_state, done = self.replay_buffer.sample(args.batch_size)
            done = 1-done  # 0 if done (so multiply done = 0 below), 1 else
            
            # train critic
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()
            current_Q = self.critic(state, action)
            
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.c_trainLoss.append(critic_loss.item())
            
            # train actor
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
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

def main():
    agent = DDPG(state_dim, action_dim, max_action)
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
        state = env.reset()
        for i in range(args.batch_size):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
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

            # a timestep
            for t in count():
                action = agent.select_action(state)
                # action = action + exploration_noise
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                
                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval : env.render()
                agent.replay_buffer.push((state, action, reward, next_state, np.float(done)))
                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
                
            total_step += step+1
            agent.sum_rewards.append(total_reward)
            agent.update()  # train agent at the end of the episode
            print("Episode {}, length: {} timesteps, reward: {:.1f}, moving average reward: {:.1f}, time used: {:.1f}".format(
                    i, step, total_reward, np.mean(agent.sum_rewards[-10:]), time.time() - start_time))
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f
            
            if i % args.log_interval == 0:
                agent.save()
        agent.plot()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()