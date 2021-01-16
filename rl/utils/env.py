from typing import Callable, Iterable
from itertools import cycle

from gym import Env
import torch
import numpy as np

def repeat_action(env: Env, action: int, repeats: int):
    reward = 0.0
    for _ in range(repeats):
        next_state, r, done, info = env.step(action)
        reward += r
        if done:
            break
    return next_state, reward, done, info


class ParallelEnv:
    def __init__(self, env_gen_fn: Callable[[], Env], no_envs: int, no_repeats: int=1):
        self.envs = np.array([env_gen_fn() for _ in range(no_envs)])
        self.no_repeats = no_repeats

    def step(self, actions: Iterable[int]):
        assert actions.shape[0] == len(self.envs), "Must provide an action for each env"

        states, rewards, dones = [], [], []

        for action, env in zip(actions, self.envs):
            next_state, reward, done, _ = repeat_action(env, int(action), self.no_repeats)
            states.append(next_state); rewards.append(reward); dones.append(done)

        return torch.as_tensor(np.stack(states), dtype=torch.float), torch.tensor(rewards), torch.tensor(dones), {}

    def reset(self, mask: Iterable[bool]=None):
        envs = self.envs if mask is None else self.envs[mask.numpy()]

        states = []
        for env in envs:
            states.append(env.reset())

        return torch.as_tensor(np.stack(states), dtype=torch.float)
