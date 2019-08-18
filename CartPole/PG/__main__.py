import gym

import matplotlib.pyplot as plt

import torch
import torch.cuda as cuda
from torch.optim import Adam
from torch.nn import MSELoss
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from reinforce.network import Policy

L = 0.99    # Discount
render = False
C = 10     # If render=True, how often to render the episode.
E = 1000    # Episodes
policy_LR = 1e-3

env = gym.make('CartPole-v1')
available_actions = [0, 1]

policy = Policy(); policy.to(device)
policy_opt = Adam(policy.parameters(), lr=policy_LR)

final_rewards = []

for e in range(E):
    
    states = []
    actions = []
    rewards = []
    action_dists = []

    state = env.reset()

    tot_reward = 0.0
    
    while True:

        action_dist = policy(torch.as_tensor(state, device=device, dtype=torch.float).view(1, 4)).view(-1)
        action = choices(available_actions, weights=action_dist, k=1)[0]

        next_state, reward, done, _ = env.step(action)
        if render and e % C == 0:
            env.render()

        tot_reward += reward

        states.append(state)
        actions.append(action)
        action_dists.append(action_dist)
        rewards.append(reward)
        
        state = next_state

        if done:
            break

    n = len(rewards)
    G = torch.empty(n)
    G[-1] = rewards[-1]
    for i in range(2, n+1):
        G[-i] = rewards[-i] + L * G[-(i-1)]
        
    policy_loss = - (G * torch.log(torch.stack(action_dists)[torch.arange(0,n), actions])).mean()
    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    print("Episode {} - Reward {}".format(e, tot_reward))
    final_rewards.append(tot_reward)

avg = torch.as_tensor(final_rewards, dtype=torch.float).cumsum(dim=0) / torch.arange(1, E+1, dtype=torch.float)

plt.plot(range(1, E+1), final_rewards, label="Episodic reward")
plt.plot(range(1, E+1), list(avg), label="Average reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()