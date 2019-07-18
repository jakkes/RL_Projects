from DQN.replay import Replay
from DQN.networks import Net

import torch
from torch.optim import Adam
from torch.nn import MSELoss

import gym
env = gym.make("LunarLander-v2")

from copy import deepcopy
from random import randint, random

C = 15      # How often to update target network
M = 100000   # Replay memory size
TRAIN_START = 1000
EPS = 1.0  # Epsilon greedy
MIN_EPS = 0.0001
DECAY_EPS = 0.995
Lambda = 0.99   # Discount
B = 64      # Batch size

actions = [0, 1, 2, 3]

replay = Replay(M)
network = Net()
target_network = deepcopy(network)

opt = Adam(network.parameters(), lr=1e-3)
loss_fn = MSELoss()

EPS = EPS / DECAY_EPS**10

counter = 0
while True:
    counter += 1
    
    prev_state = None
    state = torch.as_tensor(env.reset())

    if counter % C == 0:
        target = deepcopy(network)
    
    print("Episode {} - Epsilon {} - Q-value {}".format(
        counter, EPS, network(state.view(1, 8)).max().item()
    ))

    done = 1

    while done == 1:
        
        if random() < EPS or replay.count() < TRAIN_START:
            action = randint(0, 3)
        else:
            action = network(state.view((1, ) + state.shape)).argmax().item()

        prev_state = state
        state, reward, done, _ = env.step(action)
        state = torch.as_tensor(state)
        done = 1 if not done else 0

        replay.add( (prev_state, state, action, reward, done) )

        if replay.count() > TRAIN_START:

            states = torch.empty(B, 8)
            next_states = torch.empty(B, 8)
            all_actions = torch.empty(B, dtype=torch.long)
            rewards = torch.empty(B)
            dones = torch.empty(B)

            for i in range(B):
                s, sp, a, r, d = replay.get_random()
                states[i,:] = s
                next_states[i,:] = sp
                all_actions[i] = a
                rewards[i] = r
                dones[i] = d

            Q = network(states)         # type: torch.Tensor
            Q = Q[torch.arange(0,B), all_actions]
            
            target_Q = rewards + Lambda * dones * torch.max(target_network(next_states), dim=1).values.detach()

            loss = loss_fn(Q, target_Q)
            opt.zero_grad()
            loss.backward()
            opt.step()


        env.render()
        if done == 0:
            break
            
    EPS = EPS * DECAY_EPS
    if EPS < MIN_EPS:
        EPS = MIN_EPS
