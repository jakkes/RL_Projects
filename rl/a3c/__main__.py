from time import sleep

import gym
import torch
from torch import optim, nn, Tensor

import matplotlib.pyplot as plt

from . import A3Cagent, A3Cconfig
from rl.utils.env import repeat_action

if __name__ == "__main__":
    env_gen = lambda: gym.make("LunarLander-v2")
    value_net_gen = lambda: nn.Sequential(
        nn.Linear(8, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 1)
    )

    policy_net_gen = lambda: nn.Sequential(
        nn.Linear(8, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 4), nn.Softmax(dim=1)
    )

    config = A3Cconfig(
        env_gen=env_gen,
        value_net_gen=value_net_gen,
        policy_net_gen=policy_net_gen,
        optimizer=optim.Adam,
        optimizer_params={'lr': 1e-4},
        actors=8,
        batchsize=64,
        action_repeats=2,
        discount=0.995
    )

    agent = A3Cagent(config)
    agent.start_training()

    env: gym.Env = env_gen()

    rewards = []
    start_values = []
    
    while True:
        sleep(3)
        state = torch.as_tensor(env.reset()).view(1, -1)
        done = False
        total_reward = 0
        start_value = agent.get_values(state).item()
        start_values.append(start_value)
        while not done:
            action = agent.get_actions(state).item()

            next_state, reward, done, _ = repeat_action(env, action, config.action_repeats)
            next_state = torch.as_tensor(next_state).view(1, -1)
            total_reward += reward
            state = next_state
            env.render()
        print(f"Reward: {round(total_reward, 1)} - Start value: {round(start_value, 1)}")
        rewards.append(total_reward)
        plt.cla()
        plt.plot(rewards, label="Episode rewards")
        plt.plot(start_values, label="Start values")
        plt.legend()
        plt.pause(0.001)