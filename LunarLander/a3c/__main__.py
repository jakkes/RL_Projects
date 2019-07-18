import torch
from torch.optim import Adam

import gym

from threading import Thread
from random import choices

from a3c.networks import Actor, Critic

L = 0.95
N = 16
N_TO_RENDER = 1
BETA = 0.001

actor = Actor()
actor_opt = Adam(actor.parameters(), lr=1e-6)

critic = Critic()
critic_opt = Adam(critic.parameters(), lr=1e-5)

available_actions = [0, 1, 2, 3]

def run():
    envs = [gym.make("LunarLander-v2") for _ in range(N)]

    prev_states = torch.empty(N, 8)
    states = torch.empty(N, 8)
    dones = torch.zeros(N)
    I = torch.ones(N, dtype=torch.float)

    while running:
        
        for i in range(N):
            if dones[i] == 0:
                prev_states[i, :] = torch.as_tensor(envs[i].reset())
                dones[i] = 1
                I[i] = 1.0
            else:
                prev_states[i, :] = states[i, :]
        
        action_probabilities = actor(prev_states)
        actions = torch.empty(N, dtype=torch.long)
        rewards = torch.empty(N)

        for i in range(N):
            actions[i] = choices(available_actions, weights=action_probabilities[i, :], k=1)[0]
            s, r, d, _ = envs[i].step(actions[i].item())
            dones[i] = 1 if not d else 0
            states[i, :] = torch.as_tensor(s)
            rewards[i] = r

        td_targets = rewards + L * dones * critic(states).detach()

        delta = td_targets - critic(prev_states)
        loss1 = (I * delta.pow(2)).mean()
        critic_opt.zero_grad()
        loss1.backward()
        critic_opt.step()

        delta = delta.detach()
        loss2 = -(I * delta * torch.log(action_probabilities[torch.arange(0,N), actions])).mean() + BETA * torch.mean(torch.log(action_probabilities) * action_probabilities)
        actor_opt.zero_grad()
        loss2.backward()
        actor_opt.step()
        
        I *= L

        for i in range(N_TO_RENDER):
            envs[i].render()

    for env in envs:
        env.close()

    

running = False
run_thread = Thread(target=run)

while True:
    try:
        print()
        print("S - start/stop")
        print("R - restart")
        print("N<int> - Change number of envs")
        print("R<int> - Change number of renders")

        choice = input("Write your choice: ").lower()   # type: str

        if choice == "s":
            running = not running
            if running:
                run_thread = Thread(target=run)
                run_thread.start()
            else:
                run_thread.join()
        elif choice == "r":
            if not running:
                running = True
                run_thread = Thread(target=run)
                run_thread.start()
            else:
                running = False
                run_thread.join()
                running = True
                run_thread = Thread(target=run)
                run_thread.start()
        elif choice.startswith("n"):
            N = int(choice[1:])
            if N < N_TO_RENDER:
                N = N_TO_RENDER
        elif choice.startswith("r"):
            N_TO_RENDER = int(choice[1:])

    except Exception:
        continue
