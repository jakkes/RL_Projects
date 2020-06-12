import os
import json
import asyncio
from random import random, randrange
from typing import List
from time import sleep
from threading import Thread

import ray
import gym
import numpy as np

import torch
from torch import nn, optim, Tensor

from replay import PrioritizedReplayBuffer


BATCH_SIZE = 256
DISCOUNT = 0.99
TRAINING_START = 10000
STEP_LIMIT = 500
STEPS = 2
ALPHA = 0.6
BETA = 0.0
TARGET_UPDATE_FREQ = 10
LR = 1e-4
REPLAY_SIZE = int(2**17)
POLICY_UPDATE_FREQ = 50
N = 8


QNet = lambda: nn.Sequential(
    nn.Linear(8, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 4)
)


@ray.remote(num_cpus=2)
class Replay:
    def __init__(self):
        self.replay: PrioritizedReplayBuffer = PrioritizedReplayBuffer(REPLAY_SIZE)

    def sample(self):
        return self.replay.sample(BATCH_SIZE)

    def update_weights(self, weights):
        self.replay.update_weights(weights)

    def add_sample(self, sample, weight):
        self.replay.add(sample, weight)

    def add_multiple(self, samples, weights):
        self.replay.add_multiple(samples, weights)

    def get_size(self):
        return self.replay.get_size()


@ray.remote
class Worker:
    def __init__(self, replay: Replay, epsilon: float, render=False):
        
        print(epsilon)

        self.epsilon = epsilon
        self.replay = replay

        self.Q: nn.Module = QNet()
        self.Q.requires_grad_(False)
        self.env: gym.Env = gym.make("LunarLander-v2")

        self.render = render

    def update_policy(self, state_dict):
        self.Q.load_state_dict(state_dict)

    async def run(self):

        send_task = None

        while True:

            state = torch.as_tensor(self.env.reset()).unsqueeze_(0)
            Q: Tensor = self.Q(state).squeeze_(0)
            done = False

            steps = 0
            tot_reward = 0
            samples = []
            weights = []
            while not done:
                await asyncio.sleep(0.005)
                steps += 1

                if random() < self.epsilon:
                    action = randrange(4)
                else:
                    action = Q.argmax().item()

                reward = 0
                for _ in range(STEPS):
                    next_state, r, done, _ = self.env.step(action)
                    reward += r
                    if done:
                        break

                tot_reward += reward
                next_state = torch.as_tensor(next_state).unsqueeze_(0)
                nextQ = self.Q(next_state).squeeze_(0)

                if steps > STEP_LIMIT:
                    done = True

                weight = reward + DISCOUNT * nextQ.max() - Q[action] if not done else reward - Q[action]
                weight.abs_().pow_(ALPHA)

                samples.append((state, action, reward, done, next_state))
                weights.append(weight.item())

                Q = nextQ
                state = next_state

                # if self.render:
                #     self.env.render()
            
            
            if send_task is not None:
                await send_task
            send_task = self.replay.add_multiple.remote(
                np.array(samples, dtype=object),
                np.array(weights)
            )

            if self.render:
                print(tot_reward)


@ray.remote(num_cpus=4)
class Learner:
    def __init__(self, replay: Replay, workers: List[Worker]):
        
        self.replay: Replay = replay
        self.workers: List[Worker] = workers

        self.Q: nn.Module = QNet()
        self.targetQ: nn.Module = QNet()
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.targetQ.requires_grad_(False)
        self.opt = optim.Adam(self.Q.parameters(), lr=LR)

    async def run(self):

        while await self.replay.get_size.remote() < TRAINING_START:
            await asyncio.sleep(1.0)

        print("Starting")

        states = torch.empty(BATCH_SIZE, 8)
        next_states = torch.empty(BATCH_SIZE, 8)
        actions = torch.empty(BATCH_SIZE, dtype=torch.long)
        rewards = torch.empty(BATCH_SIZE)
        not_dones = torch.empty(BATCH_SIZE, dtype=torch.bool)

        arange = torch.arange(BATCH_SIZE)
        
        steps = 0
        update_weight_task = None
        policy_update_task = None
        while True:
            steps += 1

            if steps % TARGET_UPDATE_FREQ == 0:
                self.targetQ.load_state_dict(self.Q.state_dict())

            if steps % POLICY_UPDATE_FREQ == 0:
                if policy_update_task is not None:
                    await policy_update_task
                policy_update_task = asyncio.gather(*[worker.update_policy.remote(self.Q.state_dict()) for worker in self.workers])

            if update_weight_task is not None:
                await update_weight_task
            samples, prio = await self.replay.sample.remote()

            w = (1 / REPLAY_SIZE / torch.as_tensor(prio)).pow_(BETA)
            w /= w.max()
            for i in range(BATCH_SIZE):
                states[i] = samples[i][0]
                actions[i] = samples[i][1]
                rewards[i] = samples[i][2]
                not_dones[i] = not samples[i][3]
                next_states[i] = samples[i][4]

            Q = self.Q(states)
            targetQ = self.targetQ(next_states)
            
            with torch.no_grad():
                next_greedy_actions = self.Q(next_states).argmax(1)
            
            td_error = rewards + not_dones * DISCOUNT * targetQ[arange, next_greedy_actions] - Q[arange, actions]

            new_weights = td_error.detach().abs().pow(ALPHA).numpy()

            update_weight_task = self.replay.update_weights.remote(new_weights)

            loss = (w * td_error).pow(2).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


if __name__ == "__main__":
    
    ray.init()

    replay = Replay.remote()

    epsilons = np.power(0.4, 1 + np.arange(N) * 3 / (N-1)) if N > 1 else [0.1]
    workers = [Worker.remote(replay, epsilon, render=epsilon == epsilons[-1]) for epsilon in epsilons]
    worker_run_ids = [worker.run.remote() for worker in workers]

    learner = Learner.remote(replay, workers)
    learner_run_id = learner.run.remote()

    while True:
        sleep(1.0)
