import gym
from threading import Thread
from time import sleep

import torch
from torch import nn, optim

import matplotlib.pyplot as plt

from utils.env import ParallelEnv

from . import PPOAgent, PPOConfig

ACTORS = 8
TRAIN_STEPS = 16 # per actor
EPOCHS = 10
STEPS = 2   # steps to repeat action

class Training(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.mean_reward = 0.0

    def run(self):
        val_net = lambda: nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        pol_net = lambda: nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 4), nn.Softmax(dim=1)
        )

        config = PPOConfig(
            epsilon=0.15,
            policy_net_gen=pol_net,
            value_net_gen=val_net,
            optimizer=optim.Adam,
            optimizer_params={'lr': 5e-5},
            discount=0.99,
            gae_discount=0.95
        )

        agent = PPOAgent(config)

        env_gen_fn = lambda: gym.make("LunarLander-v2")
        env = ParallelEnv(env_gen_fn, ACTORS, no_repeats=STEPS)
        start_states = env.reset()
        state_shape = start_states.shape[1:]

        states = torch.empty(ACTORS, TRAIN_STEPS+1, *state_shape)
        actions = torch.empty(ACTORS, TRAIN_STEPS, dtype=torch.long)
        rewards = torch.empty(ACTORS, TRAIN_STEPS)
        not_dones = torch.empty(ACTORS, TRAIN_STEPS, dtype=torch.bool)
        states[:, -1] = start_states

        total_rewards = torch.zeros(ACTORS)
        while True:

            states[:, 0] = states[:, -1]
            for k in range(TRAIN_STEPS):
                
                with torch.no_grad():
                    actions[:, k] = agent.get_actions(states[:, k])
                
                s, r, d, _ = env.step(actions[:, k])
                states[:, k+1] = s
                rewards[:, k] = r
                not_dones[:, k] = ~d

                total_rewards += r

                if any(d):
                    states[d, k+1] = env.reset(d)
                    self.mean_reward += 0.01 * (total_rewards[d].mean().item() - self.mean_reward)
                    total_rewards[d] = 0.0

            agent.train_step(states, actions, rewards, not_dones, EPOCHS)


if __name__ == "__main__":
    trainer = Training()
    trainer.start()

    rewards = []

    while True:
        rewards.append(trainer.mean_reward)
        plt.cla()
        plt.plot(rewards)
        plt.pause(0.001)
        sleep(10.0)