from typing import Callable, Dict

import torch
from torch import Tensor

@torch.jit.script
def _loss2(old_values: Tensor, V: Tensor, Vtarget: Tensor, old_probs: Tensor, new_probs: Tensor, epsilon: Tensor):
    A = Vtarget - V
    
    policy_mask = (((new_probs - old_probs) / old_probs).abs_() < epsilon).detach_()
    value_mask = (((V - old_values) / old_values.abs()).abs_() < epsilon).detach_()

    policy_loss = - torch.where(policy_mask, A.detach() * new_probs.log(), torch.zeros_like(new_probs)).mean()
    value_loss = torch.where(value_mask, A.pow(2), torch.zeros_like(A)).mean()

    return policy_loss + value_loss

@torch.jit.script
def _loss(old_values: Tensor, V: Tensor, Vtarget: Tensor, old_probs: Tensor, new_probs: Tensor, epsilon: Tensor):
    A = (Vtarget - V).detach()
    pr = new_probs / old_probs
    clipped = pr.clamp(1-epsilon.item(), 1+epsilon.item())
    pr = pr * A; clipped = clipped * A
    policy_loss = - torch.min(clipped, pr).mean()

    V = ((V - old_values) / old_values.abs()).clamp_max_(epsilon.item())
    V *= old_values.abs()
    V += old_values
    value_loss = (Vtarget - V).pow_(2).div_(2).mean()

    return policy_loss + value_loss

class PPOConfig:
    def __init__(self, 
            epsilon: float=None,
            policy_net_gen: Callable[[], torch.nn.Module]=None,
            value_net_gen: Callable[[], torch.nn.Module]=None,
            optimizer: torch.optim.Optimizer=None,
            optimizer_params: Dict=None,
            discount: float=None
        ):
        self.epsilon: Tensor = torch.tensor(epsilon)
        self.value_net_gen: Callable[[], torch.nn.Module] = value_net_gen
        self.policy_net_gen: Callable[[], torch.nn.Module] = policy_net_gen
        self.optimizer: torch.optim.Optimizer = optimizer
        self.optimizer_params: Dict = optimizer_params
        self.discount: Tensor = torch.tensor(discount)


class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config: PPOConfig = config

        self.policy_net = config.policy_net_gen()
        self.value_net = config.value_net_gen()
        self.optimizer: torch.optim.Optimizer = config.optimizer(list(self.value_net.parameters()) + list(self.policy_net.parameters()), **config.optimizer_params)

        self._states = None
        self._actions = None
        self._rewards = None
        self._not_dones = None
        self._next_states = None
        self._reset_memory()

    def _reset_memory(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._not_dones = []
        self._next_states = []

    def get_actions(self, states: torch.Tensor):
        with torch.no_grad():
            action_distributions: torch.Tensor = self.policy_net(states)
        r = torch.rand(states.shape[0]).view(-1, 1)
        actions = (r > action_distributions.cumsum(1)).sum(1)
        return actions

    def observe(self, state, action, reward, not_done, next_state):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._not_dones.append(not_done)
        self._next_states.append(next_state)

    def train_step(self, epochs, batchsize):
        
        states = torch.stack(self._states)
        actions = torch.tensor(self._actions, dtype=torch.long)
        rewards = torch.tensor(self._rewards, dtype=torch.float)
        not_dones = torch.tensor(self._not_dones, dtype=torch.bool)
        next_states = torch.stack(self._next_states)
        self._reset_memory()
        
        n = states.shape[0]

        with torch.no_grad():
            old_probs = self.policy_net(states)[torch.arange(n), actions]
            old_values = self.value_net(states)

        for _ in range(epochs):
            indices = torch.randperm(n)
            for b in range(0, n, batchsize):
                batch = indices[b:b+batchsize]
                new_probs = self.policy_net(states[batch])[torch.arange(batch.shape[0]), actions[batch]]
                with torch.no_grad():
                    Vtarget = rewards[batch] + self.config.discount * not_dones[batch] * self.value_net(next_states[batch]).view(-1)
                V = self.value_net(states[batch]).view(-1)

                loss = _loss(old_values[batch], V, Vtarget, old_probs[batch], new_probs, self.config.epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()