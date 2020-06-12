from typing import List, Dict, Any

import torch
from torch import Tensor, nn, cuda
from torch.optim import Optimizer

from .network import RainbowNet
from .replay import PrioritizedReplayBuffer

class RainbowConfig:
    def __init__(self, state_dim: int, action_dim: int, pre_stream_hidden_layer_sizes: List[int],
                    value_stream_hidden_layer_sizes: List[int], advantage_stream_hidden_layer_sizes: List[int], 
                    device: torch.device, no_atoms: int, Vmax: float, Vmin: float, std_init: float, 
                    optimizer: Optimizer, optimizer_params: Dict[str, Any], n_steps: int, discount: float,
                    replay_capacity: int, batchsize: int, beta_start: float, beta_end: float, beta_t_start: int, beta_t_end: int,
                    alpha: float, target_update_freq: int):
        

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

class RainbowAgent:
    def __init__(self, config: RainbowConfig):
        self.config: RainbowConfig = config

        self.Qnet: nn.Module = None
        self.Tnet: nn.Module = None
        self.opt: Optimizer
        self.replay = PrioritizedReplayBuffer(config)
        self.z = torch.linspace(config.Vmin, config.Vmax, steps=config.no_atoms)
        self.dz = self.z[1] - self.z[0]

        self._build_networks()
        self._build_optimizer()

        self._states = torch.zeros(config.n_steps, config.state_dim)
        self._actions = torch.zeros(config.n_steps, dtype=torch.long)
        self._rewards = torch.zeros(config.n_steps, config.n_steps)
        self._discount_vector = torch.pow(config.discount, torch.arange(config.n_steps, dtype=torch.float))
        self._n_step = 0
        self._index_vector = torch.arange(config.n_steps)
        self._start_adding = False

        self._beta_coeff = (config.beta_end - config.beta_start) / (config.beta_t_end - config.beta_t_start)

        self.train_steps = 0

        self._batch_vec = torch.arange(self.config.batchsize)

    def _build_networks(self):
        self.Qnet = RainbowNet(self.config).to(self.config.device)
        self.Tnet = RainbowNet(self.config).requires_grad_(False).eval().to(self.config.device)
        self._target_update()
    
    def _target_update(self):
        self.Tnet.load_state_dict(self.Qnet.state_dict())

    def _build_optimizer(self):
        self.opt = self.config.optimizer(self.Qnet.parameters(), **self.config.optimizer_params)

    def observe(self, state: Tensor, action: int, reward: float, not_done: bool, next_state: Tensor):
        self._states[self._n_step] = state
        self._actions[self._n_step] = action
        self._rewards[(self._n_step + self._index_vector) % self.config.n_steps, self._index_vector] = reward

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
                    (self._rewards[self._n_step] * self._discount_vector).sum(),
                    True,
                    next_state
                )
                self._rewards[self._n_step].fill_(0.0)

    def get_actions(self, states) -> Tensor:
        d = self.Qnet(states)
        expected_value = torch.sum(d * self.z.view(1, 1, -1), dim=2)
        return expected_value.argmax(dim=1)

    def train_step(self):
        states, actions, rewards, not_dones, next_states, sample_prob = self.replay.sample(self.config.batchsize)
        beta = min(max(self._beta_coeff * (self.train_steps - self.config.beta_t_start) + self.config.beta_start, self.config.beta_start), self.config.beta_end)

        current_distribution: Tensor = self.Qnet(states)[self._batch_vec, actions, :]
        target_distribution = self.Tnet(next_states)
        with torch.no_grad():
            next_greedy_actions = self.get_actions(next_states)
        next_distribution = target_distribution[self._batch_vec, next_greedy_actions, :]

        m = torch.zeros(self.config.batchsize, self.config.no_atoms)
        projection = (rewards.view(-1, 1) + not_dones.view(-1, 1) * self.config.discount * self.z.view(1, -1)).clamp_(self.config.Vmin, self.config.Vmax)
        b = (projection - self.config.Vmin) / self.dz
        
        l = b.floor().to(torch.long); u = b.ceil().to(torch.long)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.config.no_atoms - 1)) * (l == u)] += 1

        for batch in range(self.config.batchsize):
            m[batch].put_(l[batch], next_distribution[batch] * (u[batch] - b[batch]), accumulate=True)
            m[batch].put_(u[batch], next_distribution[batch] * (b[batch] - l[batch]), accumulate=True)

        loss = - (m * current_distribution.add_(1e-6).log()).sum(1)

        w = (1.0 / self.replay.get_size() / sample_prob) ** beta
        w /= w.max()

        self.opt.zero_grad()
        (w * loss).mean().backward()
        self.opt.step()

        self.replay.update_weights(loss.detach().pow_(self.config.alpha))

        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._target_update()
