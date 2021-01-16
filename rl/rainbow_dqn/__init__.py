from typing import List, Dict, Any

import torch
from torch import Tensor, nn, cuda
from torch.optim import Optimizer
import torch.nn.functional as F

from .network import RainbowNet
from .replay import Replay, PrioritizedReplayBuffer, UniformReplayBuffer


class RainbowConfig:
    def __init__(self, state_dim: int=None, action_dim: int=None, pre_stream_hidden_layer_sizes: List[int]=None,
                 value_stream_hidden_layer_sizes: List[int]=None, advantage_stream_hidden_layer_sizes: List[int]=None,
                 device: torch.device=None, no_atoms: int=None, Vmax: float=None, Vmin: float=None, std_init: float=None,
                 optimizer: Optimizer=None, optimizer_params: Dict[str, Any]=None, n_steps: int=None, discount: float=None,
                 replay_capacity: int=None, batchsize: int=None, beta_start: float=None, beta_end: float=None, 
                 beta_t_start: int=None, beta_t_end: int=None, alpha: float=None, target_update_freq: int=None,
                 use_noisy: bool=None, use_double: bool=None, use_prioritized_replay: bool=None, use_dueling: bool=None,
                 use_distributional: bool=None, gradient_clip: float=-1):

        self.use_noisy: bool = use_noisy
        self.use_double: bool = use_double
        self.use_prioritized_replay: bool = use_prioritized_replay
        self.use_dueling: bool = use_dueling
        self.use_distributional: bool = use_distributional

        self.device: torch.device = device

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.pre_stream_hidden_layer_sizes: List[int] = pre_stream_hidden_layer_sizes
        self.value_stream_hidden_layer_sizes: List[int] = value_stream_hidden_layer_sizes
        self.advantage_stream_hidden_layer_sizes: List[int] = advantage_stream_hidden_layer_sizes
        self.std_init: float = std_init

        self.optimizer: Optimizer = optimizer
        self.optimizer_params: Dict[str, Any] = optimizer_params

        self.no_atoms: int = no_atoms
        self.Vmax: float = Vmax
        self.Vmin: float = Vmin

        self.n_steps: int = n_steps
        self.discount: float = discount
        self.replay_capacity: int = replay_capacity
        self.batchsize: int = batchsize
        self.target_update_freq: int = target_update_freq

        self.beta_start: float = beta_start
        self.beta_end: float = beta_end
        self.beta_t_start: float = beta_t_start
        self.beta_t_end: float = beta_t_end
        self.alpha: float = alpha

        self.gradient_clip: float = gradient_clip


class RainbowAgent:
    def __init__(self, config: RainbowConfig):
        self.config: RainbowConfig = config

        self.Qnet: nn.Module = None
        self.Tnet: nn.Module = None
        self.opt: Optimizer
        self.replay: Replay = PrioritizedReplayBuffer(config) if config.use_prioritized_replay else UniformReplayBuffer(config)
        if config.use_distributional:
            self.z = torch.linspace(
                config.Vmin, config.Vmax, steps=config.no_atoms)
            self.dz = self.z[1] - self.z[0]

        self._build_networks()
        self._build_optimizer()

        self._states = torch.zeros(config.n_steps, config.state_dim)
        self._actions = torch.zeros(config.n_steps, dtype=torch.long)
        self._rewards = torch.zeros(config.n_steps, config.n_steps)
        self._discount_vector = torch.pow(
            config.discount, torch.arange(config.n_steps, dtype=torch.float))
        self._n_step = 0
        self._index_vector = torch.arange(config.n_steps)
        self._start_adding = False

        if config.use_prioritized_replay:
            self._beta_coeff = (config.beta_end - config.beta_start) / \
                (config.beta_t_end - config.beta_t_start)

        self.train_steps = 0

        self._batch_vec = torch.arange(self.config.batchsize)

    def _build_networks(self):
        self.Qnet = RainbowNet(self.config).to(self.config.device)
        self.Tnet = RainbowNet(self.config).requires_grad_(
            False).eval().to(self.config.device)
        self._target_update()

    def _target_update(self):
        self.Tnet.load_state_dict(self.Qnet.state_dict())

    def _build_optimizer(self):
        self.opt = self.config.optimizer(
            self.Qnet.parameters(), **self.config.optimizer_params)

    def observe(self, state: Tensor, action: int, reward: float, not_done: bool, next_state: Tensor):
        self._states[self._n_step] = state
        self._actions[self._n_step] = action
        self._rewards[(self._n_step + self._index_vector) %
                      self.config.n_steps, self._index_vector] = reward

        if not not_done:    # if done
            num_to_report = self.config.n_steps if self._start_adding else self._n_step
            for i in range(num_to_report):
                self.replay.add(
                    self._states[i],
                    self._actions[i],
                    (self._rewards[i] * self._discount_vector).sum(),
                    False,
                    next_state
                )
            self._rewards.fill_(0)
            self._n_step = 0
            self._start_adding = False
        else:
            self._n_step += 1
            if self._n_step >= self.config.n_steps:
                self._start_adding = True
                self._n_step = 0

            if self._start_adding:
                self.replay.add(
                    self._states[self._n_step],
                    self._actions[self._n_step],
                    (self._rewards[self._n_step] *
                     self._discount_vector).sum(),
                    True,
                    next_state
                )
                self._rewards[self._n_step].fill_(0.0)

    def get_qvalues(self, states):
        if self.config.use_distributional:
            d = self.Qnet(states)
            return torch.sum(d * self.z.view(1, 1, -1), dim=2)
        else:
            return self.Qnet(states)

    def get_actions(self, states) -> Tensor:
        return self._get_actions(states, self.Qnet)

    def get_target_actions(self, states) -> Tensor:
        return self._get_actions(states, self.Tnet)

    def _get_actions(self, states, network):
        if self.config.use_distributional:
            d = network(states)
            expected_value = torch.sum(d * self.z.view(1, 1, -1), dim=2)
            return expected_value.argmax(dim=1)
        else:
            return network(states).argmax(dim=1)

    def _get_distributional_loss(self, states, actions, rewards, not_dones, next_states):
        current_distribution: Tensor = self.Qnet(
            states)[self._batch_vec, actions, :]
        target_distribution = self.Tnet(next_states)
        with torch.no_grad():
            if self.config.use_double:
                next_greedy_actions = self.get_actions(next_states)
            else:
                next_greedy_actions = self.get_target_actions(next_states)
        next_distribution = target_distribution[self._batch_vec,
                                                next_greedy_actions, :]

        m = torch.zeros(self.config.batchsize, self.config.no_atoms)
        projection = (rewards.view(-1, 1) + not_dones.view(-1, 1) * self.config.discount *
                      self.z.view(1, -1)).clamp_(self.config.Vmin, self.config.Vmax)
        b = (projection - self.config.Vmin) / self.dz

        l = b.floor().to(torch.long)
        u = b.ceil().to(torch.long)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.config.no_atoms - 1)) * (l == u)] += 1

        for batch in range(self.config.batchsize):
            m[batch].put_(l[batch], next_distribution[batch] *
                          (u[batch] - b[batch]), accumulate=True)
            m[batch].put_(u[batch], next_distribution[batch] *
                          (b[batch] - l[batch]), accumulate=True)

        return - (m * current_distribution.add_(1e-6).log()).sum(1)

    def _get_td_error(self, states, actions, rewards, not_dones, next_states):
        current_q_values = self.Qnet(states)[self._batch_vec, actions]
        if self.config.use_double:
            with torch.no_grad():
                next_greedy_actions = self.get_actions(next_states)
                target_values = self.Tnet(next_states)[self._batch_vec, next_greedy_actions]
        else:
            target_values = self.Tnet(next_states).max(dim=1).values
        return rewards + not_dones * self.config.discount * target_values - current_q_values

    def train_step(self):
        states, actions, rewards, not_dones, next_states, sample_prob = self.replay.sample(
            self.config.batchsize)
        
        if self.config.use_distributional:
            loss = self._get_distributional_loss(states, actions, rewards, not_dones, next_states)
        else:
            tderror = self._get_td_error(states, actions, rewards, not_dones, next_states)

        self.opt.zero_grad()
        if self.config.use_prioritized_replay:
            beta = min(max(self._beta_coeff * (self.train_steps - self.config.beta_t_start) +
                        self.config.beta_start, self.config.beta_start), self.config.beta_end)
            w = (1.0 / self.replay.get_size() / sample_prob) ** beta
            w /= w.max()
            if self.config.use_distributional:
                (w * loss).mean().backward()
                self.replay.update_weights(loss.detach().pow_(self.config.alpha))
            else:
                self.replay.update_weights(tderror.abs().detach().pow(self.config.alpha))
                (w * tderror.pow(2)).mean().backward()
        else:
            if self.config.use_distributional:
                loss.mean().backward()
            else:
                tderror.pow(2).mean().backward()
        
        if self.config.gradient_clip > 0:
            params: Tensor
            for params in self.Qnet.parameters():
                params.grad.clamp_(-self.config.gradient_clip, self.config.gradient_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._target_update()
