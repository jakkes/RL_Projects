from typing import Callable, Dict, Tuple, Union

import torch
from torch import Tensor, LongTensor, Size
import numpy as np
import gym

from utils.random import choice
from utils.replay import UniformSequenceBuffer

import MuZero.mcts as mcts

@torch.jit.script
def _N_to_policy(N, T):
    N = N.pow(1.0 / T)
    return N / N.sum(1, keepdim=True)

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
        action_dim: int=None,
        optimizer_class: torch.optim.Optimizer=None,
        optimizer_params: Dict=None
    ):
    
        self.representation_net_gen: Callable[[], torch.nn.Module] = representation_net_gen
        self.prediction_net_gen: Callable[[], torch.nn.Module] = prediction_net_gen
        self.dynamics_net_gen: Callable[[], torch.nn.Module] = dynamics_net_gen
        self.c1: float = c1
        self.c2: float = c2
        self.simulations: int = simulations
        self.discount: float = torch.tensor(discount)
        self.policy_temperature: float = torch.tensor(policy_temperature)
        self.replay_capacity: int = replay_capacity
        self.state_shape: Union[Size, Tuple] = state_shape
        self.action_dim: int = action_dim
        self.optimizer_class: torch.optim.Optimizer = optimizer_class
        self.optimizer_params: Dict = optimizer_params

class MuZeroAgent:
    def __init__(self, config: MuZeroConfig):
        self.config: MuZeroConfig = config

        self.replay = UniformSequenceBuffer(config.replay_capacity, 3)
        
        self.representation_net = config.representation_net_gen()
        self.prediction_net = config.prediction_net_gen()
        self.dynamics_net = config.dynamics_net_gen()

        self.optimizer = self.config.optimizer_class(
            list(self.representation_net.parameters()) + list(self.prediction_net.parameters()) + list(self.dynamics_net.parameters()),
            **self.config.optimizer_params
        )

    def _requires_grad(self, val):
        self.representation_net.requires_grad_(val)
        self.dynamics_net.requires_grad_(val)
        self.prediction_net.requires_grad_(val)

    def get_actions(self, states: Tensor) -> LongTensor:
        self._requires_grad(False)
        states = self.representation_net(states)
        _, _, N = mcts.run_mcts(states, self.config.simulations, self)
        policy = _N_to_policy(N, self.config.policy_temperature)
        return choice(policy)

    def observe(self, states, actions, rewards):
        self.replay.add((states, actions, rewards))

    def train_step(self, batchsize: int, unroll_steps: int):
        if batchsize > self.replay.get_size():
            raise ValueError(f"Cannot sample {batchsize} samples from buffer of size {self.replay.get_size()}")

        self._requires_grad(True)
        sequences, _ = self.replay.sample(batchsize)
        state_shape = sequences[0, 0].shape[1:]

        start_states = torch.zeros(batchsize, *state_shape)
        true_actions = torch.zeros(batchsize, unroll_steps, dtype=torch.long)
        true_rewards = torch.zeros(batchsize, unroll_steps)

        for i in range(batchsize):
            l = sequences[i, 1].shape[0]
            j = np.random.randint(l)
            k = min(unroll_steps, l-j)
            start_states[i] = sequences[i, 0][j]
            true_actions[i, :k] = sequences[i, 1][j:j+unroll_steps]
            true_actions[i, k:] = true_actions[i, k-1]
            true_rewards[i, :k] = sequences[i, 2][j:j+unroll_steps]

        root_states = self.representation_net(start_states)
        state_shape = root_states.shape[1:]

        states = torch.zeros(batchsize, unroll_steps+1, *state_shape)
        rewards = torch.zeros(batchsize, unroll_steps, 1)

        states[:, 0] = root_states
        for k in range(unroll_steps):
            states[:, k+1], rewards[:, k] = self.dynamics_net(states[:, k], true_actions[:, k])
        
        priors, values = self.prediction_net(states.view(-1, *state_shape))
        priors = priors.view(batchsize, unroll_steps+1, -1)
        values = values.view(batchsize, unroll_steps+1)


        self._requires_grad(False)
        P, Q, N = mcts.run_mcts(states.detach().view(-1, *state_shape), self.config.simulations, self)
        mcts_policy = _N_to_policy(N, self.config.policy_temperature).view(batchsize, unroll_steps+1, -1)
        mcts_value = (Q.view(batchsize, unroll_steps+1, -1) * mcts_policy).sum(-1)

        ploss = - (mcts_policy * priors.log()).sum(-1).mean()
        rloss = (rewards - true_rewards).pow(2).mean()
        vloss = (values - mcts_value).pow(2).mean()
        loss = ploss + rloss + vloss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
