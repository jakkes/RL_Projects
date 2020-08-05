from typing import Callable, Dict, Tuple, Union

import torch
from torch import Tensor, LongTensor, Size
import numpy as np
import gym

from utils.random import choice
from utils.replay import UniformReplayBuffer

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
        self.replay = UniformReplayBuffer(config.state_shape, config.replay_capacity)

    def _requires_grad(self, val):
        self.representation_net.requires_grad_(val)
        self.dynamics_net.requires_grad_(val)
        self.prediction_net.requires_grad_(val)

    def get_actions(self, states: Tensor) -> LongTensor:
        self._requires_grad(False)
        _, _, N = mcts.run_mcts(states, self.config.simulations, self)
        N = N[torch.arange(states.shape[0]), 1:1+self.config.action_dim]
        N = N.pow(1.0 / self.config.policy_temperature)
        policy = N / N.sum(1, keepdim=True)
        return choice(policy)

    def observe(self, state, action, reward, not_done, next_state):
        self.replay.add(state, action, reward, not_done, next_state)

    def train_step(self, batchsize: int):
        if batchsize > self.replay.get_size():
            raise ValueError(f"Cannot sample {batchsize} samples from buffer of size {self.replay.get_size()}")

        self._requires_grad(True)

        states, actions, rewards, not_dones, next_states, _ = self.replay.sample(batchsize)
        hidden_states = self.representation_net(states)
        with torch.no_grad():
            next_hidden_states = self.representation_net(next_states)
        next_hidden_states[~not_dones] = hidden_states[~not_dones].detach()

        predicted_priors, predicted_values = self.prediction_net(hidden_states)
        with torch.no_grad():
            _, next_values = self.prediction_net(next_hidden_states)

        predicted_next_hidden_states, predicted_rewards = self.dynamics_net(hidden_states, actions)
        
        policies = torch.zeros_like(predicted_priors)
        target_values = torch.zeros_like(predicted_values)
        for i, h in enumerate(hidden_states):
            node = mcts.run_mcts(h, self)
            policies[i] = self.get_policy(node)
            values[i] = rewards[i] + not_dones[i] * self.config.discount * node.Q[actions[i]]

        reward_loss = (predicted_rewards - rewards).pow_(2).mean()
        policy_loss = (policies * predicted_priors.log_()).sum(dim=1).mean()
