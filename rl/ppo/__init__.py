from typing import Callable, Dict

import torch
from torch import Tensor, nn


@torch.jit.script
def _policy_loss(A: Tensor, old_probs: Tensor, new_probs: Tensor, epsilon: Tensor):
    pr = new_probs / old_probs
    clipped = pr.clamp(1-epsilon.item(), 1+epsilon.item())
    pr = pr * A; clipped = clipped * A
    return -torch.min(clipped, pr).mean()


@torch.jit.script
def _compute_deltas(rewards, V, not_dones, discount):
    deltas = torch.empty(V.shape[0], V.shape[1] - 1)
    for k in torch.arange(deltas.shape[1]-1, -1, -1):
        deltas[:, k] = rewards[:, k] + discount * not_dones[:, k] * V[:, k+1].detach() - V[:, k]
    return deltas


@torch.jit.script
def _compute_advantages(deltas, not_dones, discount, gae_discount):
    
    d = discount*gae_discount
    
    A = torch.empty_like(deltas)
    A[:, -1] = deltas[:, -1]
    for k in torch.arange(A.shape[1]-2, -1, -1):
        A[:, k] = deltas[:, k] + d * not_dones[:, k] * A[:, k+1]
    return A


class PPOConfig:
    def __init__(self, 
            epsilon: float=None,
            policy_net_gen: Callable[[], torch.nn.Module]=None,
            value_net_gen: Callable[[], torch.nn.Module]=None,
            optimizer: torch.optim.Optimizer=None,
            optimizer_params: Dict=None,
            discount: float=None,
            gae_discount: float=None
        ):
        self.epsilon: Tensor = torch.tensor(epsilon)
        self.value_net_gen: Callable[[], torch.nn.Module] = value_net_gen
        self.policy_net_gen: Callable[[], torch.nn.Module] = policy_net_gen
        self.optimizer: torch.optim.Optimizer = optimizer
        self.optimizer_params: Dict = optimizer_params
        self.discount: Tensor = torch.tensor(discount)
        self.gae_discount: Tensor = torch.tensor(gae_discount)


class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config: PPOConfig = config

        self.policy_net = config.policy_net_gen()
        self.value_net = config.value_net_gen()
        self.optimizer: torch.optim.Optimizer = config.optimizer(list(self.value_net.parameters()) + list(self.policy_net.parameters()), **config.optimizer_params)


    def get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            action_distributions: torch.Tensor = self.policy_net(states)
        r = torch.rand(states.shape[0]).view(-1, 1)
        actions = (r > action_distributions.cumsum(1)).sum(1)
        return actions


    def train_step(self, states, actions, rewards, not_dones, epochs):
        
        n = states.shape[0]     # number of actors
        b = states.shape[1]     # steps per actor
        nbvec = torch.arange(n*(b-1))
        state_shape = states.shape[2:]
        
        with torch.no_grad():
            old_probs = self.policy_net(states[:, :-1].reshape(-1, *state_shape))[nbvec, actions.view(-1)].view(n, b-1)
        
        for _ in range(epochs):
            V = self.value_net(states.view(-1, *state_shape)).view(n, b)
            d = _compute_deltas(rewards, V, not_dones, self.config.discount)
            A = _compute_advantages(d.detach(), not_dones, self.config.discount, self.config.gae_discount)
            
            new_probs = self.policy_net(states[:, :-1].reshape(-1, *state_shape))[nbvec, actions.view(-1)].view(n, b-1)

            loss = _policy_loss(A, old_probs, new_probs, self.config.epsilon) + d.pow(2).div_(2.0).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
