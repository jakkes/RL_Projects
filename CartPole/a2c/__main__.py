import gym

import torch
import torch.cuda as cuda
from torch.optim import Adam
from torch.nn import MSELoss
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from a2c.network import Actor, Critic

L = 0.99    # Discount

env = gym.make('CartPole-v0')
available_actions = [0, 1]

actor = Actor(); actor.to(device)
critic = Critic(); critic.to(device)

actor_opt = Adam(actor.parameters(), lr=1e-2)
critic_opt = Adam(actor.parameters(), lr=2e-1)

mse_loss = MSELoss()

for e in range(10000):
    
    current_state = torch.as_tensor(env.reset(), dtype=torch.float, device=device).view(1, 4)      # Reset environment
    next_state = None
    done = False

    tot_reward = 0
    initial_value = critic(current_state).item()

    while not done:
        action_probabilities = actor(current_state).view(-1)                            # Get policy in current state                                 
        action = choices(available_actions, weights=action_probabilities, k=1)[0]       # Choose action according to policy

        next_state, reward, done, _ = env.step(action)                                  # Take step
        next_state = torch.as_tensor(next_state, dtype=torch.float, device=device).view(1, 4)          # Make state a torch tensor
        tot_reward += reward                                                            # Store total reward

        if e % 10 == 0:
            env.render()

        td_target = reward
        if not done:
            td_target += L * critic(next_state).detach()
                                                                                        
        V = critic(current_state)

        critic_loss = - (td_target - V.detach()) * V
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        
        actor_loss = - (td_target - V.detach()) * torch.log(action_probabilities[action])
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        current_state = next_state

    print("Episode {} - Reward {} - Initial value {}".format(
        e, tot_reward, initial_value
    ))