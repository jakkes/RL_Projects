from time import sleep
from random import choices

import gym

import torch
from torch import nn, optim, Tensor
from torch.multiprocessing import Process, Queue
from multiprocessing.connection import Connection

DISCOUNT = 0.99
STEPS = 2
N = 8
B = 64
EPISODE_LENGTH = 500

class Worker(Process):
    def __init__(self, conn: Queue):
        super().__init__(daemon=True)

        self.conn: Queue = conn
        
    def run(self):
        
        env = gym.make("LunarLander-v2")
        
        actor = self.conn.get()
        critic = self.conn.get()
        opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

        steps = 0
        loss = torch.tensor(0.0)
        while True:
            
            state = torch.as_tensor(env.reset()).view(1, -1)
            done = False

            episode_steps = 0
            while not done:
                episode_steps += 1
                steps += 1
                
                action_probabilities: Tensor = actor(state).view(-1)
                action = int(torch.sum(action_probabilities.cumsum(0) < torch.rand((1, ))))

                reward = 0
                for _ in range(STEPS):
                    next_state, r, done, _ = env.step(action)
                    reward += r
                    if done:
                        break
                next_state = torch.as_tensor(next_state).view(1, -1)

                if episode_steps > EPISODE_LENGTH:
                    done = True

                with torch.no_grad():
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + DISCOUNT * critic(next_state)[0, 0]

                delta = td_target - critic(state)[0, 0]

                loss += delta.pow(2) - delta.detach() * torch.log(action_probabilities[action])

                if steps % B == 0:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss = 0

                state = next_state



if __name__ == "__main__":
    
    actor = nn.Sequential(nn.Linear(8, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 4), nn.Softmax(dim=1))
    actor.share_memory()
    critic = nn.Sequential(nn.Linear(8, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
    critic.share_memory()

    queues = [Queue() for _ in range(N)]
    for q in queues:
        q.put(actor); q.put(critic)

    workers = [Worker(q) for q in queues]
    for worker in workers:
        worker.start()

    env = gym.make("LunarLander-v2")
    
    while True:

        done = False
        state = torch.as_tensor(env.reset()).view(1, -1)
        tot_reward = 0
        episode_steps = 0
        while not done:
            episode_steps += 1
            action_probabilities = actor(state).view(-1)
            action = int(torch.sum(action_probabilities.cumsum(0) < torch.rand((1, ))))

            reward = 0
            for _ in range(STEPS):
                state, r, done, _ = env.step(action)
                reward += r
                if done:
                    break
            state = torch.as_tensor(state).view(1, -1)
            tot_reward += reward

            if episode_steps > EPISODE_LENGTH:
                done = True

            env.render()

        print(tot_reward)