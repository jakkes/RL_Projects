import torch
from torch.optim import Adam
from torch import nn

import gym
env = gym.make("LunarLander-v2")

from random import choices

L = 0.99

actor = nn.Sequential(
    nn.Linear(8, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 4), nn.Softmax(dim=1)
)
actor_opt = Adam(actor.parameters(), lr=1e-4)

critic = nn.Sequential(
    nn.Linear(8, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
critic_opt = Adam(critic.parameters(), lr=1e-4)

loss1 = 0
loss2 = 0

steps = 0
for episode in range(10000):
    
    state = torch.as_tensor(env.reset())
    prev_state = None
    
    e = 0

    done = False
    tot_reward = 0
    while not done:
        e += 1
        steps += 1
        
        prev_state = state
        action_probabilities = actor(state.view(1, -1)).view(-1)
        action = choices([0, 1, 2, 3], weights=action_probabilities, k=1)[0]
        
        state, reward, done, _ = env.step(action)
        state = torch.as_tensor(state)
        tot_reward += reward
        
        if e >= 400:
            done = True

        if episode % 100 == 0:
            env.render()

        td_target = reward
        if not done:
            td_target += L * critic(state.view(1,-1)).detach()

        delta = td_target - critic(prev_state.view(1,-1))
        loss1 += delta.pow(2) / 400

        delta = delta.detach()
        loss2 += -(delta * torch.log(actor(prev_state.view(1, -1))[0, action])) / 400
        
        if steps % 400 == 0:
            critic_opt.zero_grad()
            loss1.backward()
            critic_opt.step()
            loss1 = 0

            actor_opt.zero_grad()
            loss2.backward()
            actor_opt.step()
            loss2 = 0
        
    print(tot_reward)