import gym
import matplotlib.pyplot as plt
import torch
import torch.cuda as cuda
from torch.optim import Adam, SGD
device = 'cuda' if cuda.is_available() else 'cpu'

from random import choices

from neat.network import Policy

render = True
C = 10      # How often to render the episode.
E = 1000   # Episodes
N = 100
eps = 1e-2
eps_decay = 0.99

if __name__ == '__main__':
    envs = [gym.make('CartPole-v1') for _ in range(N)]
    policies = [Policy() for _ in range(N)]
    available_actions = [0, 1]

    final_rewards = []
    states = torch.empty(N, 4, device=device, dtype=torch.float)

    for e in range(E):
        eps = eps * eps_decay
        for i in range(N):
            states[i, :] = torch.as_tensor(envs[i].reset(), device=device, dtype=torch.float)

        rewards = [0] * N
        dones = [False] * N

        for i in range(N):
            while not dones[i]:
                s, r, d, _ = envs[i].step(policies[i](states[i].view(1, -1)).argmax().item())
                states[i, :] = torch.as_tensor(s, device=device, dtype=torch.float)
                rewards[i] += r
                dones[i] = d

        final_rewards.append(max(rewards))
        best_net = max(range(N), key=lambda i: rewards[i])

        if render and e % C == 0:
            en = envs[0]
            s = torch.as_tensor(en.reset(), device=device, dtype=torch.float)
            d = False
            en.render()
            while not d:
                a = policies[best_net](s.view(1,-1)).argmax().item()
                s, _, d, _ = en.step(a)
                s = torch.as_tensor(s, device=device, dtype=torch.float)
                en.render()

        state_dict = policies[best_net].state_dict()
        for policy in policies:     # type: Policy
            policy.load_state_dict(state_dict)
            
        with torch.no_grad():
            for policy in policies:
                for param in policy.parameters():
                    param.add_(torch.randn_like(param) * eps)

        print("Episode {} - Reward {} - Best network {} - EPS {}".format(
            e, final_rewards[-1], best_net, eps
        ))

    avg = torch.as_tensor(final_rewards, dtype=torch.float).cumsum(dim=0) / torch.arange(1, E+1, dtype=torch.float)

    plt.plot(range(1, E+1), final_rewards, label="Episodic reward")
    plt.plot(range(1, E+1), list(avg), label="Average reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()