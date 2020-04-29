from random import random, randrange
from typing import List
from time import sleep

import gym

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.multiprocessing import Process, Queue, SimpleQueue
# torch.multiprocessing.set_sharing_strategy('file_system')

from replay import PrioritizedReplayBuffer


BATCH_SIZE = 128
DISCOUNT = 0.99
TRAINING_START = 1000
STEP_LIMIT = 500
STEPS = 2
ALPHA = 0.6
BETA = 0.5
TARGET_UPDATE_FREQ = 10
LR = 1e-4
REPLAY_SIZE = int(2**15)
POLICY_UPDATE_FREQ = 20
N = 8


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.a = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.v(x) + self.a(x) - self.a(x).mean(dim=1, keepdim=True)


class Replay(Process):
    def __init__(self, l2r: SimpleQueue, r2l: SimpleQueue, sampleQueue: Queue):
        super().__init__(daemon=True)

        self.l2r: SimpleQueue = l2r
        self.r2l: SimpleQueue = r2l
        self.sq: Queue = sampleQueue
        self.replay: PrioritizedReplayBuffer = None

    def run(self):
        did_start_feeding = False
        self.replay = PrioritizedReplayBuffer(REPLAY_SIZE)

        while True:
            
            if not did_start_feeding and self.replay.get_size() > TRAINING_START:
                did_start_feeding = True
                self.r2l.put(self.replay.sample(BATCH_SIZE))

            if not self.l2r.empty() and did_start_feeding:
                self.replay.update_weights(self.l2r.get())
                self.r2l.put(self.replay.sample(BATCH_SIZE))

            samples = []
            weights = []
            while not self.sq.empty() and len(samples) < 100:
                sample, weight = self.sq.get()
                samples.append(sample)
                weights.append(weight)

            if len(samples) > 0:
                self.replay.add_multiple(np.array(samples, dtype=object), np.array(weights))


class Worker(Process):
    def __init__(self, sampleQueue: Queue, policyQueue: Queue, epsilon: float):
        super().__init__(daemon=True)

        print(epsilon)

        self.epsilon = epsilon
        self.sampleQueue: Queue = sampleQueue
        self.policyQueue: Queue = policyQueue
        self.render = False

        self.Q: nn.Module = None
        self.env: gym.Env = None

    def run(self):
        self.Q = QNet()
        self.Q.requires_grad_(False)
        self.Q.load_state_dict(self.policyQueue.get())

        self.env = gym.make("LunarLander-v2")

        while True:

            new_policy = None
            while not self.policyQueue.empty():
                new_policy = self.policyQueue.get()
            if new_policy is not None:
                self.Q.load_state_dict(new_policy)

            state = torch.as_tensor(self.env.reset()).unsqueeze_(0)
            Q: Tensor = self.Q(state).squeeze_(0)
            done = False

            steps = 0
            tot_reward = 0
            while not done:
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

                self.sampleQueue.put((
                    (state, action, reward, done, next_state),
                    weight.item()
                ))

                Q = nextQ
                state = next_state

                if self.render:
                    self.env.render()
            if self.render:
                print(tot_reward)


class Learner(Process):
    def __init__(self, r2l: SimpleQueue, l2r: SimpleQueue, policyQueues: List[Queue]):
        super().__init__(daemon=True)

        self.policyQueues: List[Queue] = policyQueues
        self.r2l: SimpleQueue = r2l
        self.l2r: SimpleQueue = l2r

        self.Q: nn.Module = None
        self.targetQ: nn.Module = None

    def run(self):

        self.Q = QNet()
        self.targetQ = QNet()
        self.targetQ.load_state_dict(self.Q.state_dict())
        self.targetQ.requires_grad_(False)

        opt = optim.Adam(self.Q.parameters(), lr=LR)

        for q in self.policyQueues:
            q.put(self.Q.state_dict())

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
                for q in self.policyQueues:
                    q.put(self.Q.state_dict())

            samples, prio = self.r2l.get()
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
            self.l2r.put(new_weights)

            loss = (w * td_error).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    
    l2r = SimpleQueue()
    r2l = SimpleQueue()
    sampleQueue = Queue(1000)

    replay = Replay(l2r, r2l, sampleQueue)

    policyQueues = [Queue(10) for _ in range(N)]
    epsilons = np.linspace(0.4, 0.001, num=N)
    workers = [Worker(sampleQueue, policyQueues[i], epsilons[i]) for i in range(N)]
    workers[-1].render = True

    learner = Learner(r2l, l2r, policyQueues)

    replay.start()
    for worker in workers:
        worker.start()
    learner.start()

    while True:
        sleep(1.0)