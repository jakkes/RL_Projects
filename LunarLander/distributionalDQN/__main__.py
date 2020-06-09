from .replay import Replay
from .network import Net
import matplotlib.pyplot as plt

import torch
from torch import nn, optim, Tensor

import gym
env = gym.make("LunarLander-v2")

from copy import deepcopy
from random import randrange, random

C = 10      # How often to update target network
M = 20000   # Replay memory size
TRAIN_START = 1000
EPS = 1.0  # Epsilon greedy
MIN_EPS = 0.01
DECAY_EPS = 0.995
Lambda = 0.99   # Discount
B = 32      # Batch size
Vmax = 300
Vmin = -300
Natoms = 71

z = torch.linspace(Vmin, Vmax, Natoms)
dz = z[1] - z[0]

replay = Replay(M)
network = Net(z)
target_network = Net(z)
target_network.requires_grad_(False)

opt = optim.Adam(network.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()

mean_reward = 0.0
mean_rewards = [0.0]

steps = 0
for episode in range(10000):
    
    state = torch.as_tensor(env.reset())
    with torch.no_grad():
        start_value = network.getQ(state.unsqueeze(0)).max().item()

    done = False
    episode_steps = 0
    total_reward = 0
    while not done:
        
        if random() < EPS:
            action = randrange(4)
        else:
            action = network.getQ(state.unsqueeze(0)).argmax().item()

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
            
            nextp = target_network(next_states)
            next_greedy_action = target_network.get_Q_from_p(nextp).argmax(dim=1)
            nextgreedyp = nextp[torch.arange(B), next_greedy_action, :]         # shape B x Natoms

            Tz = rewards.view(-1, 1) + Lambda * not_dones.view(-1, 1) * z.view(1, -1)
            Tz.clamp_(Vmin, Vmax)

            b = (Tz - Vmin) / dz             # shape B x Natoms
            l, u = b.floor().long(), b.ceil().long()
            
            # Fix disappearing mass
            l[(u > 0) * (l == u)] -= 1
            u[(l < (Natoms - 1)) * (l == u)] += 1

            m = torch.zeros(B, Natoms)

            offset = torch.linspace(0, ((B - 1) * Natoms), B).unsqueeze(1).expand(B, Natoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (nextgreedyp * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (nextgreedyp * (b - l.float())).view(-1))

            thisp: Tensor = network(states)[torch.arange(B), actions, :]
            
            loss = - (m * thisp.log_()).sum(dim=1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

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