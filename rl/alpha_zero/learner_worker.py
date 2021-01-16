import os
from time import time, perf_counter

from queue import Empty

import torch
from torch import nn, optim, Tensor, jit
from torch.multiprocessing import Process, Queue

from .config import AlphaZeroConfig


class LearnerWorker(Process):
    def __init__(self, network: nn.Module, optimizer: optim.Optimizer, config: AlphaZeroConfig, sample_queue: Queue, learner_logging_queue: Queue=None, save_path: str=None):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.config = config
        self.sample_queue = sample_queue
        self.learner_logging_queue = learner_logging_queue
        self.save_path = save_path

        self.last_save_time = None

    def train_step(self, states: Tensor, masks: Tensor, policies: Tensor, z: Tensor):
        p, v = self.network(states, masks)
        loggedp = torch.where(torch.isinf(p), torch.zeros_like(p), torch.log_softmax(p, dim=1))

        loss = (z - v.view(-1)).square().mean() - \
            (policies * loggedp).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learner_logging_queue is not None:
            self.learner_logging_queue.put_nowait(loss.detach())

        if perf_counter() - self.last_save_time > 600:
            self.save(states, masks)
            self.last_save_time = perf_counter()

    def save(self, states: Tensor, masks: Tensor):
        if self.save_path is None:
            return

        model = jit.trace(self.network, (states, masks))
        jit.save(model, os.path.join(self.save_path, str(int(time()))))

    def run(self):
        batch_states, batch_masks, batch_policies, batch_z = [], [], [], []
        L = 0
        self.last_save_time = perf_counter()
        while True:

            try:
                states, masks, policies, z = self.sample_queue.get(timeout=5)
                N = states.shape[0]
                while N > 0:
                    M = min(self.config.batch_size - L, N)
                    batch_states.append(states[:M]); states = states[M:]
                    batch_masks.append(masks[:M]); masks = masks[M:]
                    batch_policies.append(policies[:M]); policies = policies[M:]
                    batch_z.append(z[:M]); z = z[M:]
                    N -= M
                    L += M

                    if L >= self.config.batch_size:
                        self.train_step(torch.cat(batch_states), torch.cat(batch_masks), torch.cat(batch_policies), torch.cat(batch_z))
                        batch_states, batch_masks, batch_policies, batch_z = [], [], [], []
                        L = 0
            except Empty:
                continue
