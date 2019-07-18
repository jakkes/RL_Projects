import gym

import torch
import torch.cuda as cuda
from torch.optim import Adam, SGD
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from a3c.network import Actor, Critic

L = 0.99    # Discount
C = 10      # How often to render the episode.
E = 10000   # Episodes
N = 16      # Parallel envs
NR = 1      # Environments to render
actor_LR = 1e-3
critic_LR = 1e-3

available_actions = [0, 1]

actor = Actor(); actor.to(device)
critic = Critic(); critic.to(device)

actor_opt = Adam(actor.parameters(), lr=actor_LR)
critic_opt = Adam(critic.parameters(), lr=critic_LR)

envs = [gym.make('CartPole-v1') for _ in range(N)]

states = torch.empty(N, 4, device=device)
next_states = torch.empty(N, 4, device=device)
actions = torch.empty(N, dtype=torch.long, device=device)
rewards = torch.empty(N, device=device)
dones = torch.zeros(N, device=device)

while True:

    for i in range(N):
        if dones[i] == 0:
            states[i] = torch.as_tensor(envs[i].reset())
            print(critic(states[i].unsqueeze(dim=0)).item())
        else: 
            states[i] = next_states[i]

    action_dists = actor(states)
    
    rands = torch.rand(N, 1).expand(N, 2)
    actions = (action_dists.cumsum(dim=1) < rands).sum(dim=1).long()
    
    for i in range(N):
        sp, r, d, _ = envs[i].step(actions[i].item())
        next_states[i] = torch.as_tensor(sp, device=device)
        rewards[i] = r
        dones[i] = 0.0 if d else 1.0

        if i < NR:
            envs[i].render()

    td_target = rewards + dones * L * critic(next_states)
    V = critic(states)
    delta = td_target.detach() - V.detach()

    critic_loss = - torch.mean(delta * V)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_loss = - torch.mean(delta * torch.log(action_dists[torch.arange(0, N), actions]))
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()