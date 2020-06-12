from .replay import Replay
from .network import Net
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

import gym
env = gym.make("LunarLander-v2")

from copy import deepcopy
from random import randrange, random

C = 10      # How often to update target network
M = 20000   # Replay memory size
TRAIN_START = 5000
EPS = 1.0  # Epsilon greedy
MIN_EPS = 0.1
DECAY_EPS = 0.99
Lambda = 0.99   # Discount
B = 32      # Batch size

replay = Replay(M)
network = Net()
target_network = Net()
target_network.requires_grad_(False)

opt = optim.Adam(network.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()

mean_reward = 0.0
mean_rewards = [0.0]

steps = 0
for episode in range(10000):
    
    state = torch.as_tensor(env.reset())
    with torch.no_grad():
        start_value = network(state.unsqueeze(0)).max().item()

    done = False
    episode_steps = 0
    total_reward = 0
    while not done:
        
        if random() < EPS:
            action = randrange(4)
        else:
            action = network(state.unsqueeze(0)).argmax().item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.as_tensor(state)
        total_reward += reward

        if episode_steps >= 500:
            done = True

        replay.add(state, action, reward, not done, next_state)

        if replay.count() > TRAIN_START:

            if steps % C == 0:
                target_network.load_state_dict(network.state_dict())
                target_network.requires_grad_(False)

            states, actions, rewards, not_dones, next_states = replay.sample(B)
            Q = network(states)[torch.arange(B), actions]

            with torch.no_grad():
                next_greedy_action = network(next_states).argmax(dim=1)
            target_Q = rewards + Lambda * not_dones * target_network(next_states)[torch.arange(B), next_greedy_action]

            # target_Q = rewards + Lambda * not_dones * target_network(next_states).max(dim=1).values
            loss = loss_fn(Q, target_Q)

            opt.zero_grad()
            loss.backward()
            for param in network.parameters():
                param.grad.data.clamp_(-1, 1)
            opt.step()

            tot_norm = 0
            for p in network.parameters():
                tot_norm += p.grad.norm(2)
            print(tot_norm)

            steps += 1

        if episode % 100 == 0:
            env.render()
        episode_steps += 1

    EPS = max(EPS * DECAY_EPS, MIN_EPS)

    mean_reward += 0.1 * (total_reward - mean_reward)
    mean_rewards.append(mean_reward)

    if episode % 100 == 0:
        plt.cla()
        plt.plot(mean_rewards)
        plt.pause(0.001)

    print("""
    Episode {}
    Epsilon {}
    Q-value {}
    Reward  {}
    """.format(
        episode, EPS, start_value, mean_reward
    ))