from typing import Callable, List, Dict

import gym
import torch
from torch.multiprocessing import Queue

from .worker import Worker

class A3Cconfig:
    def __init__(self,
        env_gen: Callable[[], gym.Env]=None,
        value_net_gen: Callable[[], torch.nn.Module]=None,
        policy_net_gen: Callable[[], torch.nn.Module]=None,
        optimizer: torch.optim.Optimizer=None,
        optimizer_params: Dict=None,
        actors: int=None,
        batchsize: int=None,
        action_repeats: int=None,
        discount: float=None
    ):
        self.env_gen: Callable[[], gym.Env] = env_gen
        self.value_net_gen: Callable[[], torch.nn.Module] = value_net_gen
        self.policy_net_gen: Callable[[], torch.nn.Module] = policy_net_gen
        self.optimizer: torch.optim.Optimizer = optimizer
        self.optimizer_params: Dict = optimizer_params
        self.actors: int = actors
        self.batchsize: int = batchsize
        self.action_repeats: int = action_repeats
        self.discount: float = discount

class A3Cagent:
    def __init__(self, config: A3Cconfig):
        self.config: A3Cconfig = config

        self.policy = self.config.policy_net_gen(); self.policy.share_memory()
        self.value = self.config.value_net_gen(); self.value.share_memory()

        self._queues: List[Queue] = None
        self._workers: List[Worker] = None

    def start_training(self):
        if self._workers is not None:
            raise ValueError("Training is running already!")

        self._queues = [Queue() for _ in range(self.config.actors)]
        self._workers = [Worker(q) for q in self._queues]
        for q in self._queues:
            q.put(self.config.env_gen())
            q.put(self.policy)
            q.put(self.value)
            q.put(self.config.optimizer)
            q.put(self.config.optimizer_params)
            q.put(self.config.batchsize)
            q.put(self.config.action_repeats)
            q.put(self.config.discount)
        for worker in self._workers:
            worker.start()

    def pause_training(self):
        self._queues = None
        for worker in self._workers:
            worker.terminate()
            worker.join()
        self._workers = None
    
    def get_actions(self, states):
        with torch.no_grad():
            return torch.sum(self.policy(states).cumsum(1) < torch.rand((states.shape[0], 1)))

    def get_values(self, states):
        with torch.no_grad():
            return self.value(states).view(-1)
