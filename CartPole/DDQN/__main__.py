import gym
import matplotlib.pyplot as plt
import torch
import torch.cuda as cuda
from torch.optim import Adam
device = 'cuda' if cuda.is_available() else 'cpu'

from random import randint, random

from DDQN.network import Net
from DDQN.replay import Replay

from copy import deepcopy

render = False
C = 10      # How often to render the episode.

L = 0.99    # Discount
M = 10000    # Replay capacity  (half of the capacity is saved for the first observations. Only the second half is overrided by new observations)
U = 100     # Target update frequency   (in steps, not episodes)
E = 1000    # Episodes
LR = 1e-3   # Learning rate
T = 1000     # Start training when replay contains T samples
B = 32      # Batch size
eps = 0.1   # eps-greedy
eps_decay = 0.9999
eps_min = 0.01



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    available_actions = [0, 1]

    net1 = Net()
    net2 = Net()
    opt1 = Adam(net1.parameters(), lr=LR)
    opt2 = Adam(net2.parameters(), lr=LR)
    replay = Replay(M, (4, ), device=device)
    target_net1 = deepcopy(net1)
    target_net2 = deepcopy(net2)

    def train_once():

        def train():
            loss = torch.pow(rewards + L * dones * torch.max(target_net(next_states), dim=1).values.detach() - net(states)[torch.arange(0, B), actions], 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        states, next_states, actions, rewards, dones = replay.get_random(B)
        net = net1
        target_net = target_net2
        opt = opt1
        train()

        states, next_states, actions, rewards, dones = replay.get_random(B)
        net = net2
        target_net = target_net1
        opt = opt2
        train()
        
        



    final_rewards = []
    steps = 0

    for e in range(E):
        
        state = torch.as_tensor(env.reset(), device=device, dtype=torch.float).view(1, 4)

        tot_reward = 0.0                                # For printout
        initial_value = (net1(state).max().item() + net2(state).max().item()) / 2        # For printout
        
        while True:
            
            if steps % U == 0:
                target_net1 = deepcopy(net1)
                target_net2 = deepcopy(net2)

            if replay.count() < T or torch.rand(1) < eps:
                action = randint(0, 1)
            elif random() < 0.5:
                action = net1(state).argmax().item()
            else:
                action = net2(state).argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.as_tensor(next_state, device=device, dtype=torch.float).view(1, 4)

            replay.add(state, next_state, action, reward, done)

            train_once()

            tot_reward += reward

            eps = eps * eps_decay
            eps = eps_min if eps < eps_min else eps
            state = next_state
            if done:
                break

        print("Episode {} - Epsilon {} - Reward {} - Initial value {}- Memory size {}".format(
            e, eps, tot_reward, initial_value, replay.count()
        ))

        final_rewards.append(tot_reward)

    avg = torch.as_tensor(final_rewards, dtype=torch.float).cumsum(dim=0) / torch.arange(1, E+1, dtype=torch.float)

    plt.plot(range(1, E+1), final_rewards, label="Episodic reward")
    plt.plot(range(1, E+1), list(avg), label="Average reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()