from typing import Callable, Dict

import torch
from torch import Tensor, nn



@torch.jit.script
def _loss(V: Tensor, Vtarget: Tensor, old_probs: Tensor, new_probs: Tensor, epsilon: Tensor):
    A = (Vtarget - V).detach()
    pr = new_probs / old_probs
    clipped = pr.clamp(1-epsilon.item(), 1+epsilon.item())
    pr = pr * A; clipped = clipped * A
    policy_loss = -torch.min(clipped, pr).mean()

    value_loss = (Vtarget - V).pow_(2).div_(2).mean()

    return policy_loss + value_loss

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

    def observe(self, states: Tensor, actions: Tensor, rewards: Tensor, not_dones: Tensor, next_states: Tensor):
        self._states.append(states)
        self._actions.append(actions)
        self._rewards.append(rewards)
        self._not_dones.append(not_dones)
        self._next_states.append(next_states)

    def train_step(self, epochs, batchsize):
        states = torch.stack(self._states)
        actions = torch.stack(self._actions)
        rewards = torch.stack(self._rewards)
        not_dones = torch.stack(self._not_dones)
        next_states = torch.stack(self._next_states)
        self._reset_memory()
        
        n = states.shape[0]     # number of actors
        b = states.shape[1]     # step size per actor
        nbvec = torch.arange(n*b)
        state_shape = states.shape[2:]
        
        with torch.no_grad():
            old_probs = self.policy_net(states.view(-1, *state_shape))[nbvec, actions].view(n, b, *state_shape)
        
        advantages = torch.zeros(n, b)
        
        for i in reversed(range(b)):
            G = rewards[:, i] + self.config.discount * not_dones[:, i] * G

        for _ in range(epochs):
            indices = torch.randperm(n)
            for b in range(0, n, batchsize):
                batch = indices[b:b+batchsize]
                new_probs = self.policy_net(states[batch])[torch.arange(batch.shape[0]), actions[batch]]
                with torch.no_grad():
                    Vtarget = rewards[batch] + self.config.discount * not_dones[batch] * self.value_net(next_states[batch]).view(-1)
                V = self.value_net(states[batch]).view(-1)

                # loss = _loss(old_values[batch], V, Vtarget, old_probs[batch], new_probs, self.config.epsilon)
                loss = _loss(V, Vtarget, old_probs[batch], new_probs, self.config.epsilon)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5); nn.utils.clip_grad_norm_(self.value_net.parameters(), 5)
                self.optimizer.step()
