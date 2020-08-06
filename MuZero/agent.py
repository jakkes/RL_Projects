from typing import Callable, Dict, Tuple, Union

import torch
from torch import Tensor, LongTensor, Size
import numpy as np
import gym

from utils.random import choice
from utils.replay import UniformSequenceBuffer

import MuZero.mcts as mcts

class MuZeroConfig:
    def __init__(self,
        representation_net_gen: Callable[[], torch.nn.Module]=None,
        prediction_net_gen: Callable[[], torch.nn.Module]=None,
        dynamics_net_gen: Callable[[], torch.nn.Module]=None,
        c1: float=None,
        c2: float=None,
        simulations: int=None,
        discount: float=None,
        policy_temperature: float=None,
        replay_capacity: int=None,
        state_shape: Union[Size, Tuple]=None,
        action_dim: int=None
    ):
    
        self.representation_net_gen: Callable[[], torch.nn.Module] = representation_net_gen
        self.prediction_net_gen: Callable[[], torch.nn.Module] = prediction_net_gen
        self.dynamics_net_gen: Callable[[], torch.nn.Module] = dynamics_net_gen
        self.c1: float = c1
        self.c2: float = c2
        self.simulations: int = simulations
        self.discount: float = torch.tensor(discount)
        self.policy_temperature: float = policy_temperature
        self.replay_capacity: int = replay_capacity
        self.state_shape: Union[Size, Tuple] = state_shape
        self.action_dim: int = action_dim

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig):
        self.config: MuZeroConfig = config

        self.representation_net = config.representation_net_gen()
        self.prediction_net = config.prediction_net_gen()
        self.dynamics_net = config.dynamics_net_gen()
        self.replay = UniformSequenceBuffer(config.replay_capacity, 3)

    def _requires_grad(self, val):
        self.representation_net.requires_grad_(val)
        self.dynamics_net.requires_grad_(val)
        self.prediction_net.requires_grad_(val)

    def get_actions(self, states: Tensor) -> LongTensor:
        self._requires_grad(False)
        states = self.representation_net(states)
        _, _, N = mcts.run_mcts(states, self.config.simulations, self)
        N = N.pow(1.0 / self.config.policy_temperature)
        policy = N / N.sum(1, keepdim=True)
        return choice(policy)

    def observe(self, states, actions, rewards):
        self.replay.add((states, actions, rewards))

    def train_step(self, batchsize: int, unroll_steps: int):
        if batchsize > self.replay.get_size():
            raise ValueError(f"Cannot sample {batchsize} samples from buffer of size {self.replay.get_size()}")

        sequences, _ = self.replay.sample(batchsize)
        state_shape = sequences[0, 0].shape[1:]

        states = torch.zeros(batchsize, *state_shape)
        actions = torch.zeros(batchsize, unroll_steps, dtype=torch.long)
        rewards = torch.zeros(batchsize, unroll_steps)

        for i in range(batchsize):
            l = sequences[i, 1].shape[0]
            j = np.random.randint(l)
            states[i] = sequences[i, 0][j]
            actions[i, :min(unroll_steps, l-j)] = sequences[i, 1][j:j+unroll_steps]
            rewards[i, :min(unroll_steps, l-j)] = sequences[i, 2][j:j+unroll_steps]

        self._requires_grad(True)
        root_states = self.representation_net(states)
        self._requires_grad(False)
        P, Q, N = mcts.run_mcts(root_states, self.config.simulations, self)

        self._requires_grad(True)