import gym

import torch
import torch.cuda as cuda
from torch.optim import Adam
from torch.nn import MSELoss
device = 'cuda' if cuda.is_available() else 'cpu'

print('device:',device)

from random import choices

from reinforce.network import Policy, Baseline

L = 0.99    # Discount
C = 50
R = 100
E = 10000   # Episodes
policy_LR = 1e-2
baseline_LR = 2e-2

env = gym.make('LunarLander-v2')
available_actions = [0, 1]

policy = Policy(); policy.to(device)
baseline = Baseline(); baseline.to(device)

policy_opt = Adam(policy.parameters(), lr=policy_LR)
baseline_opt = Adam(baseline.parameters(), lr=baseline_LR)

for e in range(E):
    
    states = []
    actions = []
    rewards = []
    action_dists = []

    state = env.reset()

    tot_reward = 0.0
    initial_value = baseline(torch.as_tensor(state, device=device, dtype=torch.float).view(1, 8)).item()
    
    while True:

        action_dist = policy(torch.as_tensor(state, device=device, dtype=torch.float).view(1, 8)).view(-1)
        action = choices(available_actions, weights=action_dist, k=1)[0]

        next_state, reward, done, _ = env.step(action)
        if e % R == 0:
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

    V = baseline(torch.as_tensor(states, device=device, dtype=torch.float)).view(-1)
    delta = torch.pow(L, torch.arange(0, n).float()) * (G - V.detach())
    # delta = (G - V.detach())
    
    baseline_loss = - (delta * V).mean()
    policy_loss = - (delta * torch.log(torch.stack(action_dists)[torch.arange(0,n), actions])).mean()

    baseline_opt.zero_grad()
    baseline_loss.backward()
    baseline_opt.step()

    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    print("Episode {} - Reward {} - Initial V {}".format(e, tot_reward, initial_value))

    if e % C == 0 and e != 0:
        torch.save(policy.state_dict(), './reinforce/models/policy')
        torch.save(baseline.state_dict(), './reinforce/models/baseline')