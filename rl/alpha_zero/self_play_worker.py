import logging
from queue import Full

import numpy as np

import torch
from torch import nn
from torch.multiprocessing import Process, Queue

from rl.simulators import Simulator

from .mcts import mcts
from .config import AlphaZeroConfig


logger = logging.getLogger(__name__)


class SelfPlayWorker(Process):
    def __init__(self, simulator: Simulator, network: nn.Module, config: AlphaZeroConfig, sample_queue: Queue, episode_logging_queue: Queue = None):
        super().__init__()
        self.simulator = simulator
        self.network = network
        self.config = config
        self.sample_queue = sample_queue
        self.episode_logging_queue = episode_logging_queue

    def run_episode(self):
        states, action_masks, action_policies = [], [], []
        terminal = False
        reward = 0
        state, action_mask = self.simulator.reset()

        with torch.no_grad():
            start_prior, start_value = self.network(
                torch.as_tensor(state, dtype=torch.float).unsqueeze_(0),
                torch.as_tensor(action_mask).unsqueeze_(0)
            )
        start_value = start_value.item()
        start_prior = start_prior.softmax(dim=1).squeeze_(0)
        first_action_policy = None
        first_action = None

        root = None
        while not terminal:
            root = mcts(state, action_mask, self.simulator,
                        self.network, self.config, root_node=root, simulations=self.config.simulations)

            action_policy = root.action_policy
            if first_action_policy is None:
                first_action_policy = action_policy

            states.append(state)
            action_masks.append(action_mask)
            action_policies.append(torch.as_tensor(
                action_policy, dtype=torch.float))

            action = np.random.choice(action_mask.shape[0], p=action_policy)
            if first_action is None:
                first_action = action

            state, action_mask, reward, terminal, _ = self.simulator.step(
                state, action)
            root = root.children[action]

        if self.episode_logging_queue is not None:
            kl_div = -(first_action_policy *
                       torch.log_softmax(start_prior, dim=0).numpy()).sum()
            self.episode_logging_queue.put_nowait(
                (abs(reward), start_value, kl_div, first_action))

        states = torch.as_tensor(np.stack(states), dtype=torch.float)
        action_masks = torch.as_tensor(np.stack(action_masks))
        action_policies = torch.as_tensor(
            np.stack(action_policies), dtype=torch.float)
        z = torch.ones(states.shape[0])
        i = torch.arange(1, states.shape[0]+1, 2)
        j = torch.arange(2, states.shape[0]+1, 2)
        z[-i] *= reward
        z[-j] *= -reward

        return states, action_masks, action_policies, z

    def run(self) -> None:
        while True:
            try:
                self.sample_queue.put(self.run_episode(), timeout=5)
            except Full:
                logger.warn("Sample queue full. Skipping...")
                continue
