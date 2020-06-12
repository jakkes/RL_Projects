import os
import json
from random import random, randrange
from typing import List
from time import sleep
from threading import Thread

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Pipe
multiprocessing.set_sharing_strategy('file_system')
# from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection


import gym

import numpy as np

import torch
from torch import nn, optim, Tensor

from replay import PrioritizedReplayBuffer


BATCH_SIZE = 128
DISCOUNT = 0.99
TRAINING_START = 1000
STEP_LIMIT = 500
STEPS = 2
ALPHA = 0.1
BETA = 0.0
TARGET_UPDATE_FREQ = 10
LR = 1e-4
REPLAY_SIZE = int(2**17)
POLICY_UPDATE_FREQ = 50
N = 8

def json_to_tensor(data):
    if type(data) == str:
        data = json.loads(data)
    return torch.tensor(data.values).view(*data.shape)

def tensor_to_json(tensor: Tensor):
    return json.dumps({
        'values': tensor.view(-1).tolist(),
        'shape': list(tensor.shape)
    })

QNet = lambda: nn.Sequential(
    nn.Linear(8, 128), nn.ReLU(inplace=True), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 4)
)


class Replay(Process):
    def __init__(self, learnerPipe: Connection, samplePipes: List[Connection]):
        super().__init__(daemon=True)

        self.learnerPipe: Connection = learnerPipe
        self.replay: PrioritizedReplayBuffer = None
        self.samplePipes: List[Connection] = samplePipes

        self._pipe_thread: Thread = None

    def _pipe_reader(self):
        while True:
            read = False
            for pipe in self.samplePipes:
                while pipe.poll():
                    sample, weight = pipe.recv()
                    self.replay.add(sample, weight)
                    read = True
            if not read:
                print("Yes")
                sleep(0.1)

    def run(self):
        self.replay = PrioritizedReplayBuffer(REPLAY_SIZE)

        self._pipe_thread = Thread(target=self._pipe_reader)
        self._pipe_thread.start()

        while self.replay.get_size() < TRAINING_START:
            sleep(1.0)
        print("Starting training!")

        while True:
            self.learnerPipe.send(self.replay.sample(BATCH_SIZE))
            while not self.learnerPipe.poll():
                sleep(0.01)
            self.replay.update_weights(self.learnerPipe.recv())


class Worker(Process):
    def __init__(self, samplePipe: Connection, policyPipe: Connection, epsilon: float):
        super().__init__(daemon=True)

        print(epsilon)

        self.epsilon = epsilon
        self.samplePipe: Connection = samplePipe
        self.policyPipe: Connection = policyPipe
        self.render = False

        self.Q: nn.Module = None
        self.env: gym.Env = None

    def run(self):
        self.Q = QNet()
        self.Q.requires_grad_(False)
        self.Q.load_state_dict(self.policyPipe.recv())

        self.env = gym.make("LunarLander-v2")

        while True:

            new_policy = None
            while self.policyPipe.poll():
                new_policy = self.policyPipe.recv()
            if new_policy is not None:
                self.Q.load_state_dict(new_policy)

            state = torch.as_tensor(self.env.reset()).unsqueeze_(0)
            Q: Tensor = self.Q(state).squeeze_(0)
            done = False

            steps = 0
            tot_reward = 0
            while not done:
                sleep(0.05)
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

                self.samplePipe.send((
                    (state, action, reward, done, next_state),
                    weight.item()
                ))

                Q = nextQ
                state = next_state

                # if self.render:
                #     self.env.render()
            if self.render:
                print(tot_reward)


class Learner(Process):
    def __init__(self, replayPipe: Connection, policyPipes: List[Connection]):
        super().__init__(daemon=True)

        self.policyPipes: List[Connection] = policyPipes
        self.replayPipe: Connection = replayPipe

        self.Q: nn.Module = None
        self.targetQ: nn.Module = None

    def run(self):

        self.Q = QNet()
        self.targetQ = QNet()
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.targetQ.requires_grad_(False)

        opt = optim.Adam(self.Q.parameters(), lr=LR)

        for pipe in self.policyPipes:
            pipe.send(self.Q.state_dict())

        states = torch.empty(BATCH_SIZE, 8)
        next_states = torch.empty(BATCH_SIZE, 8)
        actions = torch.empty(BATCH_SIZE, dtype=torch.long)
        rewards = torch.empty(BATCH_SIZE)
        not_dones = torch.empty(BATCH_SIZE, dtype=torch.bool)

        arange = torch.arange(BATCH_SIZE)
        
        steps = 0
        while True:
            steps += 1

            if steps % TARGET_UPDATE_FREQ == 0:
                self.targetQ.load_state_dict(self.Q.state_dict())

            if steps % POLICY_UPDATE_FREQ == 0:
                for pipe in self.policyPipes:
                    pipe.send(self.Q.state_dict())

            while not self.replayPipe.poll():
                sleep(0.01)
            samples, prio = self.replayPipe.recv()
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
            self.replayPipe.send(new_weights)

            loss = (w * td_error).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    
    epsilons = np.power(0.4, 1 + np.arange(N) * 3 / (N-1)) if N > 1 else [0.1]
    
    learnerPipe, replayPipe = Pipe(True)
    policyPipes = [Pipe(False) for _ in range(N)]
    samplePipes = [Pipe(False) for _ in range(N)]
    
    replay = Replay(replayPipe, [x[0] for x in samplePipes])
    
    workers = [Worker(samplePipes[i][1], policyPipes[i][0], epsilons[i]) for i in range(N)]
    workers[-1].render = True

    learner = Learner(learnerPipe, [x[1] for x in policyPipes])

    replay.start()
    for worker in workers:
        worker.start()
    learner.start()

    while True:
        sleep(1.0)
