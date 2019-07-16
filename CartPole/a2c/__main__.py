import gym

import torch
from torch.optim import Adam, SGD

from random import choices

from a2c.network import Actor, Critic

L = 0.95    # Discount

env = gym.make('CartPole-v0')
available_actions = [0, 1]

actor = Actor()
critic = Critic()

actor_opt = Adam(actor.parameters(), lr=1e-3)
critic_opt = Adam(actor.parameters(), lr=1)

e = 0           # Episode count

while True:     # Run episodes forever

    e += 1
    
    current_state = torch.as_tensor(env.reset(), dtype=torch.float).view(1, 4)      # Reset environment
    next_state = None
    done = False
    I = 1.0 

    tot_critic_loss = 0                             # For print out
    tot_reward = 0
    initial_value = critic(current_state).item()


    while not done:
        action_probabilities = actor(current_state).view(-1)                            # Get policy in current state                                 
        action = choices(available_actions, weights=action_probabilities, k=1)[0]       # Choose action according to policy

        next_state, reward, done, _ = env.step(action)                                  # Take step
        next_state = torch.as_tensor(next_state, dtype=torch.float).view(1, 4)          # Make state a torch tensor
        tot_reward += reward                                                            # Store total reward

        env.render()

        td_target = reward                                                              # Compute TD target                            
        if not done:                                                                    # If not done, compute next value
            td_target += L * critic(next_state)
                                                                                        
        V = critic(current_state)
        delta = td_target - V

        critic_loss = - I * delta.detach() * V      # Compute loss. Detach delta so gradient is only taken w.r.t. V.
                                                    # Minus sign since optimizer will move in opposite direction of gradient

        critic_opt.zero_grad()                      # Reset gradients
        critic_loss.backward()                      # Compute gradients
        critic_opt.step()                           # Update parameters
        
        tot_critic_loss += critic_loss.item()       # For print out

        actor_loss = - I * delta.detach() * torch.log(action_probabilities[action])  # Compute loss. Detach and minus sign as above
        
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        I = I * L               # Update I
        current_state = next_state

    print("Episode {} - Reward {} - Initial value {} - Critic loss {}".format(
        e, tot_reward, initial_value, tot_critic_loss
    ))