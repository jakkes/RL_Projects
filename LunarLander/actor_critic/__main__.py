import torch
from torch.optim import Adam

import gym
env = gym.make("LunarLander-v2")

from random import choices

from actor_critic.networks import Actor, Critic

L = 0.99

actor = Actor()
actor.load_state_dict(torch.load('./actor_critic/actor', map_location='cpu'))
actor_opt = Adam(actor.parameters(), lr=1e-4)

critic = Critic()
critic.load_state_dict(torch.load('./actor_critic/critic', map_location='cpu'))
critic_opt = Adam(critic.parameters(), lr=2e-4)

while True:
    
    state = torch.as_tensor(env.reset())
    prev_state = None
    
    I = 1
    e = 0

    done = False
    tot_reward = 0
    while not done:
        e += 1
        
        prev_state = state
        action_probabilities = actor(state.view(1, -1)).view(-1)
        action = choices([0, 1, 2, 3], weights=action_probabilities, k=1)[0]
        
        state, reward, done, _ = env.step(action)
        state = torch.as_tensor(state)
        tot_reward += reward
        
        env.render()

        td_target = reward
        if not done:
            td_target += L * critic(state.view(1,-1)).detach()

        delta = td_target - critic(prev_state.view(1,-1))
        loss1 = I * delta.pow(2).mean()
        critic_opt.zero_grad()
        loss1.backward()
        critic_opt.step()

        delta = delta.detach()
        loss2 = -(I * delta * torch.log(actor(prev_state.view(1, -1))[0, action])).mean()
        actor_opt.zero_grad()
        loss2.backward()
        actor_opt.step()
        
        I *= L
    print(tot_reward)